import { User } from 'lucide-react'

const Header = () => {
  return (
    <header className="h-[50px] min-h-[50px] border-b border-gray-800 bg-[#0e1117] flex items-center justify-end px-6 sticky top-0 z-10">
      <div className="flex items-center gap-3 group cursor-pointer hover:bg-white/5 p-1.5 rounded transition-all">
        <div className="flex flex-col items-end">
          <span className="text-[8px] text-gray-500 font-bold tracking-widest">Operator</span>
          <span className="text-[10px] text-white font-black tracking-tight">ADMIN_CORE</span>
        </div>
        <div className="w-7 h-7 rounded-full bg-gradient-to-br from-blue-600 to-indigo-700 flex items-center justify-center text-[10px] font-bold text-white border border-blue-500/30">
          <User size={14} />
        </div>
      </div>
    </header>
  )
}

export default Header
