import { cn } from '../lib/utils'

const StatCard = ({ label, value, icon: Icon, color, accent }) => {
  return (
    <div className={cn(
      "p-6 rounded-lg border flex flex-col gap-3 relative transition-all duration-300 bg-[#161b22]",
      accent
    )}>
      <div className="flex items-center justify-between">
        <span className="text-[9px] font-bold text-gray-500 uppercase tracking-widest">{label}</span>
        <Icon size={14} className={color} />
      </div>
      <div className={cn("text-2xl font-black tracking-tighter", color)}>
        {value}
      </div>
      <div className="text-[8px] font-medium text-gray-600 uppercase">System Feedback</div>
    </div>
  )
}

export default StatCard
