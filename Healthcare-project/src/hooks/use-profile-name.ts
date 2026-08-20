"use client";

import { useEffect, useState } from "react";
import { createClient } from "@/lib/supabase/client";

export function useProfileName(fallback: string) {
  const [name, setName] = useState(fallback);

  useEffect(() => {
    let active = true;
    const supabase = createClient();

    void supabase.auth.getUser().then(async ({ data }) => {
      if (!data.user) return;

      const { data: profile } = await supabase
        .from("profiles")
        .select("full_name")
        .eq("id", data.user.id)
        .single();

      if (active && profile?.full_name) {
        setName(profile.full_name);
      }
    });

    return () => {
      active = false;
    };
  }, []);

  return name;
}
