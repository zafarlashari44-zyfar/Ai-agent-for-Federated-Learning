import React from "react";
import { cn } from "@/lib/utils";
import { Card, CardContent } from "./card";
import { LucideIcon } from "lucide-react";

interface StatCardProps {
  title: string;
  value: string | number;
  change?: string;
  changeType?: "up" | "down" | "neutral";
  icon: LucideIcon;
  iconColor?: string;
  iconBg?: string;
  className?: string;
}

export function StatCard({
  title, value, change, changeType = "neutral",
  icon: Icon, iconColor = "text-blue-600", iconBg = "bg-blue-50", className,
}: StatCardProps) {
  return (
    <Card className={cn("hover:shadow-md transition-shadow", className)}>
      <CardContent className="p-5">
        <div className="flex items-start justify-between">
          <div className="space-y-1">
            <p className="text-sm font-medium text-gray-500">{title}</p>
            <p className="text-2xl font-bold text-gray-900">{value}</p>
            {change && (
              <p className={cn(
                "text-xs font-medium",
                changeType === "up" && "text-green-600",
                changeType === "down" && "text-red-500",
                changeType === "neutral" && "text-gray-500",
              )}>
                {change}
              </p>
            )}
          </div>
          <div className={cn("p-2.5 rounded-xl", iconBg)}>
            <Icon className={cn("h-5 w-5", iconColor)} />
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
