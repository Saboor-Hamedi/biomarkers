import React from 'react'
import AnalyticView from './AnalyticView'
import { ResponsiveContainer, XAxis, YAxis, CartesianGrid, Tooltip, AreaChart, Area } from 'recharts'
import { Activity } from 'lucide-react'


const Distribution = ({ activeTab, distributionData, distributionTableData, distEntries, inputs }) => {
  return (
<div className={activeTab === 'distribution' ? 'block' : 'hidden'}>
        <AnalyticView
        title="Cohort Comparison"
        icon={Activity}
        explanation="Density plots show the frequency distribution of biomarkers in the study population. The pulsing marker indicates the current patient's position relative to the cohort."
        tableData={distributionTableData}
        columns={['Biomarker', 'Range Min', 'Range Max', 'Patient Value']}
      >
        <div className="h-[450px] overflow-y-auto custom-scrollbar pr-4 space-y-12">
          {distEntries
              .map(([key, data]) => (
                <div key={key} className="h-[180px] relative">
                  <div className="absolute top-0 left-0 text-[8px] font-black text-blue-500 mb-2">
                    {key.replace(/_/g, ' ')} Density
                  </div>
                  <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={data} margin={{ top: 20, right: 30, left: 0, bottom: 0 }}>
                      <defs>
                        <linearGradient id={`colorDensity-${key}`} x1="0" y1="0" x2="0" y2="1">
                          <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.3} />
                          <stop offset="95%" stopColor="#3b82f6" stopOpacity={0} />
                        </linearGradient>
                      </defs>
                      <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#1f2937" />
                      <XAxis dataKey="x" type="number" stroke="#4b5563" fontSize={9} />
                      <YAxis hide />
                      <Tooltip
                        contentStyle={{ backgroundColor: '#000', border: '1px solid #374151' }}
                        labelFormatter={(val) => `Value: ${val}`}
                      />
                      <Area
                        type="monotone"
                        dataKey="y"
                        stroke="#3b82f6"
                        fillOpacity={1}
                        fill={`url(#colorDensity-${key})`}
                        strokeWidth={2}
                      />

                      {/* Patient Reference Marker */}
                      {inputs[key] !== undefined && inputs[key] !== null && (
                        <Area
                          type="monotone"
                          data={[
                            { x: inputs[key], y: 0 },
                            { x: inputs[key], y: Math.max(...data.map((d) => d.y)) }
                          ]}
                          dataKey="y"
                          stroke="#ef4444"
                          strokeWidth={3}
                          strokeDasharray="5 5"
                          dot={{
                            r: 6,
                            fill: '#ef4444',
                            strokeWidth: 0,
                            className: 'animate-pulse'
                          }}
                        />
                      )}
                    </AreaChart>
                  </ResponsiveContainer>
                </div>
              ))}
        </div>
        
      </AnalyticView>
      </div>
  )
}

export default Distribution
