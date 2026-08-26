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
  Users,
  Layers
} from 'lucide-react'

import SidebarHeader from './SidebarHeader'
import SidebarFooter from './SidebarFooter'

const SidebarItem = memo(({ icon: Icon, label, active, collapsed, onClick }) => (
  <button
    onClick={onClick}
    className={cn(
      "group flex w-full items-center px-4 py-3 text-sm font-medium transition-all duration-200 border-r-2",
      collapsed ? "justify-center px-0" : "gap-3",
      active 
        ? "bg-sky-500/10 text-sky-400 border-sky-500" 
        : "text-slate-400 hover:bg-slate-800/50 hover:text-slate-200 border-transparent"
    )}
  >
    <Icon 
      className={cn("shrink-0 transition-colors h-4 w-4", active ? "text-sky-400" : "text-slate-500 group-hover:text-slate-300")} 
      strokeWidth={1.5}
    />
    {!collapsed && (
      <span className="font-['Inter',_sans-serif] tracking-wide truncate">{label}</span>
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
    { id: 'anatomy', label: '3D Anatomy', icon: Layers },
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
        "h-screen bg-[#070b12] border-r border-gray-800 flex flex-col transition-all duration-300",
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

      <div className="border-t border-gray-800 pt-2 pb-2">
        <SidebarItem
          icon={Settings}
          label="Settings"
          collapsed={collapsed}
          onClick={onOpenSettings}
        />
      </div>
    </div>
  )
})

export default Sidebar
