"use client";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { cn } from "@/lib/utils";
import { Heart, ChevronLeft } from "lucide-react";
import { useState } from "react";

interface NavItem {
  label: string;
  href: string;
  icon: React.ElementType;
  badge?: number;
}

interface SidebarProps {
  items: NavItem[];
  role: "doctor" | "admin" | "patient";
  userName?: string;
  userAvatar?: string;
}

const roleConfig = {
  doctor: { color: "bg-blue-600", light: "bg-blue-50", text: "text-blue-600", label: "Doctor Portal" },
  admin: { color: "bg-purple-600", light: "bg-purple-50", text: "text-purple-600", label: "Admin Panel" },
  patient: { color: "bg-green-600", light: "bg-green-50", text: "text-green-600", label: "Patient Portal" },
};

export function Sidebar({ items, role, userName = "User", userAvatar }: SidebarProps) {
  const pathname = usePathname();
  const [collapsed, setCollapsed] = useState(false);
  const cfg = roleConfig[role];

  return (
    <aside
      className={cn(
        "relative flex flex-col h-full border-r border-gray-100 bg-white transition-all duration-300 shrink-0",
        collapsed ? "w-16" : "w-60"
      )}
    >
      {/* Logo */}
      <div className={cn("flex items-center gap-3 p-4 border-b border-gray-100", collapsed && "justify-center")}>
        <div className={cn("h-8 w-8 rounded-lg flex items-center justify-center shrink-0", cfg.color)}>
          <Heart className="h-4 w-4 text-white" />
        </div>
        {!collapsed && (
          <div className="min-w-0">
            <p className="font-bold text-gray-900 text-sm truncate">MediCare Pro</p>
            <p className={cn("text-xs font-medium truncate", cfg.text)}>{cfg.label}</p>
          </div>
        )}
      </div>

      {/* Nav */}
      <nav className="flex-1 p-3 space-y-0.5 overflow-y-auto scrollbar-thin">
        {items.map(({ label, href, icon: Icon, badge }) => {
          const active = pathname === href || pathname.startsWith(href + "/");
          return (
            <Link
              key={href}
              href={href}
              title={collapsed ? label : undefined}
              className={cn(
                "flex items-center gap-3 px-3 py-2 rounded-lg text-sm font-medium transition-colors group",
                active
                  ? `${cfg.light} ${cfg.text}`
                  : "text-gray-600 hover:bg-gray-50 hover:text-gray-900",
                collapsed && "justify-center"
              )}
            >
              <Icon className={cn("h-4 w-4 shrink-0", active ? cfg.text : "text-gray-400 group-hover:text-gray-700")} />
              {!collapsed && (
                <>
                  <span className="flex-1 truncate">{label}</span>
                  {badge !== undefined && badge > 0 && (
                    <span className={cn("text-xs font-medium px-1.5 py-0.5 rounded-full", cfg.color, "text-white")}>
                      {badge > 99 ? "99+" : badge}
                    </span>
                  )}
                </>
              )}
            </Link>
          );
        })}
      </nav>

      {/* User */}
      {!collapsed && (
        <div className="p-3 border-t border-gray-100">
          <div className="flex items-center gap-2.5 px-2 py-2">
            {userAvatar ? (
              // eslint-disable-next-line @next/next/no-img-element
              <img
                src={userAvatar}
                alt=""
                className="h-7 w-7 rounded-full object-cover shrink-0"
              />
            ) : (
              <div className={cn("h-7 w-7 rounded-full flex items-center justify-center text-white text-xs font-semibold shrink-0", cfg.color)}>
                {userName[0]?.toUpperCase()}
              </div>
            )}
            <div className="min-w-0">
              <p className="text-sm font-medium text-gray-900 truncate">{userName}</p>
              <p className={cn("text-xs capitalize", cfg.text)}>{role}</p>
            </div>
          </div>
        </div>
      )}

      {/* Collapse toggle */}
      <button
        onClick={() => setCollapsed(!collapsed)}
        className="absolute -right-3 top-20 h-6 w-6 bg-white border border-gray-200 rounded-full flex items-center justify-center shadow-sm hover:shadow-md transition-shadow z-10"
      >
        <ChevronLeft className={cn("h-3 w-3 text-gray-500 transition-transform", collapsed && "rotate-180")} />
      </button>
    </aside>
  );
}
