import type { Database } from "@/types/database";

export type PatientProfile =
  Database["public"]["Tables"]["profiles"]["Row"];
export type PatientRecord =
  Database["public"]["Tables"]["patients"]["Row"];
export type DoctorDirectoryItem =
  Database["public"]["Functions"]["list_active_doctors"]["Returns"][number];

export interface Medication {
  name: string;
  dosage?: string;
  morning?: boolean;
  evening?: boolean;
  withFood?: boolean;
}
