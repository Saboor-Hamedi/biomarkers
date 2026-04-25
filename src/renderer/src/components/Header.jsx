import { User } from 'lucide-react'

const Header = () => {
  return (
    <header className="h-16 border-b border-gray-800 bg-[#0e1117] flex items-center justify-end px-8 sticky top-0 z-10">
      <div className="flex items-center gap-4 group cursor-pointer hover:bg-white/5 p-2 rounded-lg transition-all">
        <div className="flex flex-col items-end">
          <span className="text-[9px] text-gray-500 font-bold uppercase tracking-widest">Operator Session</span>
          <span className="text-xs text-white font-black tracking-tight italic">ADMIN_CORE_01</span>
        </div>
        <div className="w-9 h-9 rounded-full bg-gradient-to-br from-blue-600 to-indigo-700 flex items-center justify-center text-[10px] font-bold text-white border border-blue-500/30">
          <User size={18} />
        </div>
      </div>
    </header>
  )
}

export default Header
