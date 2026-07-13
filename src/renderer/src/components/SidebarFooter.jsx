import React, { memo } from 'react'
import { Settings } from 'lucide-react'
import { cn } from '../lib/utils'

const SidebarFooterItem = memo(({ icon: Icon, label, collapsed, onClick }) => (
  <button
    onClick={onClick}
    className={cn(
      "flex items-center w-full py-1.5 transition-all duration-200 group relative",
      collapsed ? "justify-center px-0" : "px-2",
      "text-gray-500 hover:bg-gray-800/40 hover:text-gray-300 rounded"
    )}
  >
    <Icon size={12} className="shrink-0 group-hover:text-blue-400" />
    {!collapsed && (
      <span className="ml-2 text-[10px] font-['Inter',_sans-serif] font-medium tracking-wide truncate">{label}</span>
    )}
  </button>
))

const SidebarFooter = memo(({ collapsed, onOpenSettings }) => (
  <div className={cn("border-t border-gray-800 h-[50px] min-h-[50px] shrink-0 flex items-center", collapsed ? "px-1 justify-center" : "px-2")}>
    <SidebarFooterItem
      icon={Settings}
      label="Settings"
      collapsed={collapsed}
      onClick={onOpenSettings}
    />
  </div>
))

export default SidebarFooter
