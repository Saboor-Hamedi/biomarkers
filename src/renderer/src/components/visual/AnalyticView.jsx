import React from 'react'
import { Info } from 'lucide-react'
import { cn } from '../../lib/utils'

const AnalyticView = ({
  title,
  icon: Icon,
  explanation,
  children,
  tableData,
  columns,
  extraAction
}) => (
  <div className="space-y-8 animate-in slide-in-from-right-4 duration-500">
    <div className="flex items-end justify-between border-b border-gray-800 pb-6">
      <div className="flex flex-col gap-2">
        <h2 className="text-xl font-black text-white tracking-tight">{title}</h2>
        <div className="flex items-center gap-2">
          <Info size={12} className="text-blue-500" />
          <p className="text-[10px] text-gray-500 font-bold leading-tight max-w-2xl">
            {explanation}
          </p>
        </div>
      </div>
      <div className="flex items-center gap-4">
        {extraAction}
      </div>
    </div>

    <div className="grid grid-cols-1 xl:grid-cols-12 gap-8 items-start">
      <div
        className={cn(
          'bg-[#0d1117] border border-gray-800 rounded-lg p-8 min-h-[500px]',
          columns && tableData ? 'xl:col-span-8' : 'xl:col-span-12'
        )}
      >
        {children}
      </div>

      {columns && tableData && (
        <div className="xl:col-span-4 h-[500px] overflow-y-auto custom-scrollbar bg-[#0d1117] border border-gray-800 rounded-lg">
          <table className="w-full text-left border-collapse relative">
            <thead className="bg-black/90 backdrop-blur-md sticky top-0 z-10">
              <tr>
                {columns.map((col) => (
                  <th
                    key={col}
                    className="p-3 text-[8px] font-black text-gray-600 tracking-widest border-b border-gray-800"
                  >
                    {col}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-800/50">
              {tableData.map((row, i) => (
                <tr key={i} className="hover:bg-white/5 transition-colors">
                  {Object.values(row).map((val, j) => (
                    <td key={j} className="p-3 text-[10px] font-mono font-bold text-gray-300">
                      {val}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  </div>
)

export default AnalyticView
