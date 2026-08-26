import { cn } from '../lib/utils'

const StatCard = ({ label, value, icon: Icon, color, accent }) => {
  return (
    <div className="group relative overflow-hidden rounded-xl border border-slate-800 bg-slate-900/40 p-4 transition-all hover:border-slate-700 hover:bg-slate-900/60">
      {/* Subtle background glow effect */}
      <div className="absolute -right-10 -top-10 h-32 w-32 rounded-full bg-sky-500/5 blur-3xl group-hover:bg-sky-500/10 transition-colors" />
      
      <div className="relative z-10 flex flex-col h-full justify-between">
        <div className="flex items-start justify-between mb-2">
          <h3 className="text-[10px] font-bold uppercase tracking-wider text-slate-400">
            {label}
          </h3>
          {Icon && <Icon className="h-4 w-4 text-slate-500" strokeWidth={1.5} />}
        </div>
        
        <div className="mt-auto">
          <div className="flex items-baseline gap-2">
            <span className={cn("text-xl font-bold tracking-tight", color || "text-slate-100")}>
              {value}
            </span>
          </div>
          <p className="mt-1 text-[9px] text-slate-500 font-medium">
            System Feedback
          </p>
        </div>
      </div>
    </div>
  )
}

export default StatCard
