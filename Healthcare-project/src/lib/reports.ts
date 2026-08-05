import { createAdminClient } from "@/lib/supabase/admin";

export async function getAgentReports() {
  const supabase = createAdminClient();

  const { data, error } = await supabase
    .from("agent_outputs")
    .select("*")
    .order("created_at", { ascending: false });

  if (error) {
    console.error("Error fetching agent_outputs:", error);
    return [];
  }

  return data ?? [];
}

export async function getAgentReportById(id: string) {
  const supabase = createAdminClient();

  const { data, error } = await supabase
    .from("agent_outputs")
    .select("*")
    .eq("id", id)
    .limit(1);

  if (error) {
    console.error("Error fetching agent_output by id:", error);
    return null;
  }

  return data?.[0] ?? null;
}