import { useState } from 'react'
import { cn } from '../lib/utils'
import { 
  ChevronLeft, 
  ChevronRight, 
  Activity,
  LayoutDashboard,
  BrainCircuit,
  Database,
  BarChart4,
  Settings,
  Target,
  Zap,
  ShieldCheck,
  Search
} from 'lucide-react'

const SidebarItem = ({ icon: Icon, label, active, collapsed, onClick }) => (
  <button
    onClick={onClick}
    className={cn(
      "flex items-center w-full px-4 py-3 transition-all duration-200 group relative",
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
      <span className="ml-4 text-[9px] font-black uppercase tracking-[0.2em] truncate">{label}</span>
    )}
  </button>
)

const Sidebar = ({ activeTab, setActiveTab }) => {
  const [collapsed, setCollapsed] = useState(false)

  const mainItems = [
    { id: 'dashboard', label: 'Overview', icon: LayoutDashboard },
    { id: 'committee', label: 'AI Committee', icon: BrainCircuit },
    { id: 'registry', label: 'Audit Registry', icon: Database },
  ]

  const analyticItems = [
    { id: 'roc', label: 'ROC Analysis', icon: Target },
    { id: 'pr', label: 'PR Dynamics', icon: Zap },
    { id: 'cm', label: 'Confusion Matrix', icon: ShieldCheck },
    { id: 'tsne', label: 'Latent Space', icon: Search },
    { id: 'importance', label: 'Biomarker Influence', icon: BarChart4 },
  ]

  return (
    <div 
      className={cn(
        "h-screen bg-[#0d1117] border-r border-gray-800 flex flex-col transition-all duration-300",
        collapsed ? "w-16" : "w-64"
      )}
    >
      <div className="flex items-center justify-between p-6 border-b border-gray-800 h-16 shrink-0">
        {!collapsed && (
          <div className="flex items-center gap-3">
            <Activity className="text-blue-600" size={20} />
            <span className="font-black text-white tracking-[0.3em] text-sm italic">FORENSIC</span>
          </div>
        )}
        <button 
          onClick={() => setCollapsed(!collapsed)}
          className="p-1.5 hover:bg-gray-800 rounded border border-gray-800 transition-colors text-gray-500"
        >
          {collapsed ? <ChevronRight size={14} /> : <ChevronLeft size={14} />}
        </button>
      </div>

      <div className="flex-1 py-4 overflow-y-auto overflow-x-hidden custom-scrollbar space-y-6">
        <div>
          {!collapsed && <p className="px-6 mb-2 text-[7px] font-black text-gray-600 uppercase tracking-[0.3em]">Main Hub</p>}
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
          {!collapsed && <p className="px-6 mb-2 text-[7px] font-black text-gray-600 uppercase tracking-[0.3em]">Deep Discovery</p>}
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

      <div className="p-4 border-t border-gray-800 shrink-0">
        <SidebarItem
          icon={Settings}
          label="Settings"
          collapsed={collapsed}
          onClick={() => {}}
          active={false}
        />
      </div>
    </div>
  )
}

export default Sidebar
