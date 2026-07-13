import React, { memo } from 'react'
import { Activity, ChevronLeft, ChevronRight } from 'lucide-react'
import { cn } from '../lib/utils'

const SidebarHeader = memo(({ collapsed, toggleCollapsed }) => (
  <div className={cn("flex items-center border-b border-gray-800 h-[50px] min-h-[50px] shrink-0", collapsed ? "justify-center p-0" : "justify-between px-4")}>
    {!collapsed && (
      <div className="flex items-center gap-2">
        {/* <Activity className="text-blue-600" size={16} /> */}
        <span className="font-black text-white tracking-[0.1em] text-xs">BIOMARKER</span>
      </div>
    )}
    <button 
      onClick={toggleCollapsed}
      className="p-1 hover:bg-gray-800 rounded border border-gray-800 transition-colors text-gray-500"
    >
      {collapsed ? <ChevronRight size={14} /> : <ChevronLeft size={14} />}
    </button>
  </div>
))

export default SidebarHeader
