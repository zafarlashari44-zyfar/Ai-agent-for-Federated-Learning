import React from "react";
import { cn } from "@/lib/utils";

interface DashboardShellProps {
  sidebar: React.ReactNode;
  topNav: React.ReactNode;
  children: React.ReactNode;
  className?: string;
}

export function DashboardShell({ sidebar, topNav, children, className }: DashboardShellProps) {
  return (
    <div className="flex h-screen overflow-hidden bg-gray-50">
      {sidebar}
      <div className="flex flex-col flex-1 min-w-0 overflow-hidden">
        {topNav}
        <main className={cn("flex-1 overflow-y-auto p-6", className)}>
          {children}
        </main>
      </div>
    </div>
  );
}
