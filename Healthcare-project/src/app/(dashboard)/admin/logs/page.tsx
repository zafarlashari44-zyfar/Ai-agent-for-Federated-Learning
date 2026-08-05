"use client";
import { useState } from "react";
import { Card, CardContent } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Button } from "@/components/ui/button";
import { Search, Download, Filter } from "lucide-react";
import { getInitials } from "@/lib/utils";

const logs = Array.from({ length: 50 }, (_, i) => ({
  id: String(i + 1),
  user: ["Dr. Sarah Johnson", "Admin", "Dr. Michael Chen", "System", "Dr. Emily Rodriguez", "Patient James Wilson"][i % 6],
  action: [
    "Created patient record", "Updated appointment status", "Generated prescription",
    "Viewed medical history", "Logged into system", "Exported patient data",
    "Modified user permissions", "Deleted appointment", "Uploaded document", "Ran system backup",
  ][i % 10],
  resource: ["Patient", "Appointment", "Prescription", "Report", "User", "System"][i % 6],
  ip: `192.168.1.${10 + (i % 50)}`,
  time: new Date(2026, 5, 9, Math.floor(i / 3) % 24, (i * 7) % 60).toISOString(),
  status: i % 15 === 0 ? "failed" : i % 8 === 0 ? "warning" : "success" as "success" | "warning" | "failed",
}));

const statusVariant = { success: "success", warning: "warning", failed: "destructive" } as const;

export default function AdminLogsPage() {
  const [search, setSearch] = useState("");
  const [resourceFilter, setResourceFilter] = useState("all");

  const filtered = logs.filter((log) => {
    const matchSearch = log.user.toLowerCase().includes(search.toLowerCase()) || log.action.toLowerCase().includes(search.toLowerCase());
    const matchResource = resourceFilter === "all" || log.resource === resourceFilter;
    return matchSearch && matchResource;
  });

  return (
    <div className="space-y-5">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-xl font-bold text-gray-900">Activity Logs</h2>
          <p className="text-sm text-gray-500">{filtered.length} entries</p>
        </div>
        <Button variant="outline" size="sm" className="gap-2"><Download className="h-4 w-4" /> Export Logs</Button>
      </div>

      <div className="flex flex-col sm:flex-row gap-3">
        <div className="relative flex-1">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-3.5 w-3.5 text-gray-400" />
          <Input placeholder="Search logs by user or action…" className="pl-8 bg-white" value={search} onChange={(e) => setSearch(e.target.value)} />
        </div>
        <Select value={resourceFilter} onValueChange={setResourceFilter}>
          <SelectTrigger className="w-36">
            <Filter className="h-3.5 w-3.5 mr-1 text-gray-400" />
            <SelectValue placeholder="Resource" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="all">All Resources</SelectItem>
            {["Patient", "Appointment", "Prescription", "Report", "User", "System"].map((r) => (
              <SelectItem key={r} value={r}>{r}</SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>

      <Card>
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="border-b border-gray-100 bg-gray-50/50">
                <th className="text-left py-3 px-5 text-xs font-semibold text-gray-500 uppercase">User</th>
                <th className="text-left py-3 px-4 text-xs font-semibold text-gray-500 uppercase">Action</th>
                <th className="text-left py-3 px-4 text-xs font-semibold text-gray-500 uppercase hidden md:table-cell">Resource</th>
                <th className="text-left py-3 px-4 text-xs font-semibold text-gray-500 uppercase hidden lg:table-cell">IP Address</th>
                <th className="text-left py-3 px-4 text-xs font-semibold text-gray-500 uppercase hidden xl:table-cell">Time</th>
                <th className="text-left py-3 px-4 text-xs font-semibold text-gray-500 uppercase">Status</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-50">
              {filtered.map((log) => (
                <tr key={log.id} className="hover:bg-gray-50/50 transition-colors">
                  <td className="py-3 px-5">
                    <div className="flex items-center gap-2.5">
                      <Avatar className="h-7 w-7">
                        <AvatarFallback className="text-xs bg-purple-100 text-purple-700">{getInitials(log.user)}</AvatarFallback>
                      </Avatar>
                      <span className="text-sm font-medium text-gray-900 truncate max-w-[120px]">{log.user}</span>
                    </div>
                  </td>
                  <td className="py-3 px-4 text-sm text-gray-600">{log.action}</td>
                  <td className="py-3 px-4 hidden md:table-cell">
                    <Badge variant="secondary" className="text-xs">{log.resource}</Badge>
                  </td>
                  <td className="py-3 px-4 hidden lg:table-cell text-sm text-gray-400 font-mono">{log.ip}</td>
                  <td className="py-3 px-4 hidden xl:table-cell text-xs text-gray-400">
                    {new Date(log.time).toLocaleString()}
                  </td>
                  <td className="py-3 px-4">
                    <Badge variant={statusVariant[log.status]}>{log.status}</Badge>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  );
}
