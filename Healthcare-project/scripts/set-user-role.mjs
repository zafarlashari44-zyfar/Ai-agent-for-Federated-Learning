import { createClient } from "@supabase/supabase-js";
import WebSocket from "ws";

const [email, role, specialization, licenseNumber] = process.argv.slice(2);
const allowedRoles = new Set(["admin", "doctor", "patient"]);

if (!email || !allowedRoles.has(role)) {
  console.error(
    "Usage: npm run user:set-role -- <email> <admin|doctor|patient> [specialization] [license-number]"
  );
  process.exit(1);
}

if (role === "doctor" && (!specialization || !licenseNumber)) {
  console.error("Doctor specialization and license number are required.");
  process.exit(1);
}

const url = process.env.NEXT_PUBLIC_SUPABASE_URL;
const serviceRoleKey = process.env.SUPABASE_SERVICE_ROLE_KEY;

if (!url || !serviceRoleKey) {
  console.error("Supabase service credentials are missing from .env.local.");
  process.exit(1);
}

const supabase = createClient(url, serviceRoleKey, {
  auth: {
    autoRefreshToken: false,
    persistSession: false,
  },
  realtime: {
    transport: WebSocket,
  },
});

let page = 1;
let user;

while (!user) {
  const { data, error } = await supabase.auth.admin.listUsers({
    page,
    perPage: 100,
  });

  if (error) {
    console.error(`Unable to list users: ${error.message}`);
    process.exit(1);
  }

  user = data.users.find(
    (candidate) => candidate.email?.toLowerCase() === email.toLowerCase()
  );

  if (user || data.users.length < 100) {
    break;
  }

  page += 1;
}

if (!user) {
  console.error(`No registered Supabase user found for ${email}.`);
  process.exit(1);
}

const { error } = await supabase.rpc("service_set_user_role", {
  target_user_id: user.id,
  new_role: role,
  doctor_specialization: specialization ?? null,
  doctor_license_number: licenseNumber ?? null,
});

if (error) {
  console.error(`Unable to update role: ${error.message}`);
  process.exit(1);
}

console.log(`Updated ${email} to the ${role} role.`);
