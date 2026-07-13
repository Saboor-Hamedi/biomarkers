import React, { useState, memo } from 'react'
import { cn } from '../lib/utils'
import { 
  ChevronLeft, 
  ChevronRight, 
  Activity,
  LayoutDashboard,
  BrainCircuit,
  Database,
  Settings,
  TrendingUp,
  BarChart2,
  Map,
  Grid,
  GitBranch,
  LineChart,
  Crosshair,
  Sliders,
  AlertTriangle,
  LayoutGrid,
  Network,
  ListTree,
  Users
} from 'lucide-react'

import SidebarHeader from './SidebarHeader'
import SidebarFooter from './SidebarFooter'

const SidebarItem = memo(({ icon: Icon, label, active, collapsed, onClick }) => (
  <button
    onClick={onClick}
    className={cn(
      "flex items-center w-full py-3 transition-all duration-200 group relative",
      collapsed ? "justify-center px-0" : "px-4",
      active 
        ? "bg-blue-600/10 text-blue-500" 
        : "text-gray-500 hover:bg-gray-800/40 hover:text-gray-300"
    )}
  >
    {active && (
      <div className="absolute left-0 top-0 bottom-0 w-1 bg-blue-600 shadow-[0_0_10px_rgba(37,99,235,0.8)]" />
    )}
    <Icon size={16} className={cn("shrink-0", active ? "text-blue-500" : "group-hover:text-blue-400")} />
    {!collapsed && (
      <span className="ml-4 text-[12px] font-['Inter',_sans-serif] font-medium tracking-wide truncate">{label}</span>
    )}
  </button>
))

const Sidebar = memo(({ activeTab, setActiveTab, onOpenSettings }) => {
  const [collapsed, setCollapsed] = useState(() => {
    const saved = localStorage.getItem('sidebarCollapsed')
    return saved ? JSON.parse(saved) : false
  })

  const toggleCollapsed = () => {
    setCollapsed((prev) => {
      const next = !prev
      localStorage.setItem('sidebarCollapsed', JSON.stringify(next))
      return next
    })
  }

  const mainItems = [
    { id: 'dashboard', label: 'Overview', icon: LayoutDashboard },
    { id: 'committee', label: 'AI Committee', icon: BrainCircuit },
    { id: 'ranking', label: 'Patient Ranking', icon: Activity },
    { id: 'registry', label: 'Audit Registry', icon: Database },
  ]

  const analyticItems = [
    { id: 'trajectory', label: 'Risk Trajectories', icon: TrendingUp },
    { id: 'shap', label: 'Biometric Radar', icon: BarChart2 },
    { id: 'boundaries', label: 'Decision Map', icon: Map },
    { id: 'heatmap', label: 'Model Heatmap', icon: Grid },
    { id: 'counterfactual', label: 'What-If Engine', icon: GitBranch },
    { id: 'roc', label: 'ROC Analysis', icon: LineChart },
    { id: 'pr', label: 'PR Dynamics', icon: Crosshair },
    { id: 'calibration', label: 'Model Calibration', icon: Sliders },
    { id: 'calibration-risk', label: 'Calibration Risk', icon: AlertTriangle },
    { id: 'cm', label: 'Confusion Matrix', icon: LayoutGrid },
    { id: 'tsne', label: 'Latent Space', icon: Network },
    { id: 'importance', label: 'Biomarker Influence', icon: ListTree },
    { id: 'distribution', label: 'Cohort Comparison', icon: Users },
  ]

  return (
    <div 
      className={cn(
        "h-screen bg-[#0d1117] border-r border-gray-800 flex flex-col transition-all duration-300",
        collapsed ? "w-16" : "w-56"
      )}
    >
      <SidebarHeader collapsed={collapsed} toggleCollapsed={toggleCollapsed} />

      <div className="flex-1 pt-0 pb-4 overflow-y-auto overflow-x-hidden space-y-4 [&::-webkit-scrollbar]:hidden [-ms-overflow-style:none] [scrollbar-width:none]">
        <div>
          {mainItems.map((item) => (
            <SidebarItem
              key={item.id}
              {...item}
              active={activeTab === item.id}
              collapsed={collapsed}
              onClick={() => setActiveTab(item.id)}
            />
          ))}
        </div>

        <div>
          {analyticItems.map((item) => (
            <SidebarItem
              key={item.id}
              {...item}
              active={activeTab === item.id}
              collapsed={collapsed}
              onClick={() => setActiveTab(item.id)}
            />
          ))}
        </div>
      </div>

      <SidebarFooter collapsed={collapsed} onOpenSettings={onOpenSettings} />
    </div>
  )
})

export default Sidebar
