import { createServerClient } from "@supabase/ssr";
import type { User } from "@supabase/supabase-js";
import { NextResponse, type NextRequest } from "next/server";
import WebSocket from "ws";

const PUBLIC_PATHS = ["/", "/login", "/register", "/forgot-password", "/reset-password"];
const SUPABASE_REQUEST_TIMEOUT_MS = 3000;

function fetchWithTimeout(input: RequestInfo | URL, init?: RequestInit) {
  const timeoutSignal = AbortSignal.timeout(SUPABASE_REQUEST_TIMEOUT_MS);
  const signal = init?.signal
    ? AbortSignal.any([init.signal, timeoutSignal])
    : timeoutSignal;

  return fetch(input, { ...init, signal });
}

export async function proxy(request: NextRequest) {
  const pathname = request.nextUrl.pathname;
  const isPublic = PUBLIC_PATHS.some(
    (path) => pathname === path || pathname.startsWith("/api/"),
  );
  const hasSupabaseSession = request.cookies
    .getAll()
    .some(({ name }) => name.startsWith("sb-") && name.includes("-auth-token"));

  // Public routes never need proxy-level authentication.
  if (isPublic) {
    return NextResponse.next({ request });
  }

  if (!hasSupabaseSession) {
    const url = request.nextUrl.clone();
    url.pathname = "/login";
    url.searchParams.set("redirectTo", pathname);
    return NextResponse.redirect(url);
  }

  let supabaseResponse = NextResponse.next({ request });

  const supabase = createServerClient(
    process.env.NEXT_PUBLIC_SUPABASE_URL!,
    process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!,
    {
      global: {
        fetch: fetchWithTimeout,
      },
      realtime: {
        transport: WebSocket as unknown as typeof globalThis.WebSocket,
      },
      cookies: {
        getAll() {
          return request.cookies.getAll();
        },
        setAll(cookiesToSet) {
          cookiesToSet.forEach(({ name, value }) => request.cookies.set(name, value));
          supabaseResponse = NextResponse.next({ request });
          cookiesToSet.forEach(({ name, value, options }) =>
            supabaseResponse.cookies.set(name, value, options)
          );
        },
      },
    }
  );

  let user: User | null = null;

  try {
    const {
      data: { user: authenticatedUser },
    } = await supabase.auth.getUser();
    user = authenticatedUser;
  } catch (error) {
    console.warn(
      "Supabase authentication is temporarily unavailable:",
      error instanceof Error ? error.message : String(error),
    );

    const url = request.nextUrl.clone();
    url.pathname = "/login";
    url.searchParams.set("redirectTo", pathname);
    return NextResponse.redirect(url);
  }

  if (!user) {
    const url = request.nextUrl.clone();
    url.pathname = "/login";
    url.searchParams.set("redirectTo", pathname);
    return NextResponse.redirect(url);
  }

  let userRole: "admin" | "doctor" | "patient" | null = null;
  let accessRequestStatus: string | null = null;

  try {
    const [profileResult, accessRequestResult] = await Promise.all([
      supabase
        .from("profiles")
        .select("role")
        .eq("id", user.id)
        .single(),
      supabase
        .from("role_requests")
        .select("status")
        .eq("user_id", user.id)
        .maybeSingle(),
    ]);

    if (profileResult.error || accessRequestResult.error) {
      throw profileResult.error || accessRequestResult.error;
    }

    userRole = profileResult.data?.role ?? "patient";
    accessRequestStatus = accessRequestResult.data?.status ?? null;
  } catch (error) {
    console.warn(
      "Supabase profile lookup is temporarily unavailable:",
      error instanceof Error ? error.message : String(error),
    );

    const url = request.nextUrl.clone();
    url.pathname = "/login";
    url.searchParams.set("redirectTo", pathname);
    return NextResponse.redirect(url);
  }

  if (accessRequestStatus && accessRequestStatus !== "approved") {
    const url = request.nextUrl.clone();
    url.pathname = "/login";
    url.searchParams.set("approval", accessRequestStatus);
    return NextResponse.redirect(url);
  }

  const requestedPortal = (["admin", "doctor", "patient"] as const).find(
    (role) => pathname === `/${role}` || pathname.startsWith(`/${role}/`),
  );

  if (user && requestedPortal && requestedPortal !== userRole) {
    const url = request.nextUrl.clone();
    url.pathname = `/${userRole}`;
    url.search = "";
    return NextResponse.redirect(url);
  }

  return supabaseResponse;
}

export const config = {
  matcher: [
    "/((?!_next/static|_next/image|favicon.ico|.*\\.(?:svg|png|jpg|jpeg|gif|webp)$).*)",
  ],
};
