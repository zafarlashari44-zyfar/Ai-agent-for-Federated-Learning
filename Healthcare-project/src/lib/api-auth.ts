import "server-only";

import { createClient } from "@/lib/supabase/server";
import type { Database } from "@/types/database";

type UserRole = Database["public"]["Enums"]["user_role"];

export type ApiRoleCheck =
  | {
      ok: true;
      userId: string;
      role: UserRole;
    }
  | {
      ok: false;
      status: 401 | 403 | 500;
      error: string;
    };

export async function requireApiRole(
  allowedRoles: UserRole[],
): Promise<ApiRoleCheck> {
  const supabase = await createClient();
  const {
    data: { user },
    error: userError,
  } = await supabase.auth.getUser();

  if (userError || !user) {
    return {
      ok: false,
      status: 401,
      error: "Authentication required.",
    };
  }

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

  if (
    profileResult.error ||
    accessRequestResult.error ||
    !profileResult.data
  ) {
    return {
      ok: false,
      status: 500,
      error: "Unable to verify the account role.",
    };
  }

  if (
    accessRequestResult.data &&
    accessRequestResult.data.status !== "approved"
  ) {
    return {
      ok: false,
      status: 403,
      error: "Administrator approval is required before using this feature.",
    };
  }

  if (!allowedRoles.includes(profileResult.data.role)) {
    return {
      ok: false,
      status: 403,
      error: "You do not have permission to use this AI feature.",
    };
  }

  return {
    ok: true,
    userId: user.id,
    role: profileResult.data.role,
  };
}
