import React from 'react'
import AnalyticView from './AnalyticView'
import { ResponsiveContainer, ScatterChart, Scatter, XAxis, YAxis, ZAxis, CartesianGrid, Tooltip, Cell } from 'recharts'
import { Search } from 'lucide-react'


const Boundaries = ({ activeTab, boundariesData }) => {
  return (
<div className={activeTab === 'boundaries' ? 'block' : 'hidden'}>
        <AnalyticView
        title="Topographic Decision Map"
        icon={Search}
        explanation="Visualizes the physical borders where the AI switches its verdict. The map charts AFP vs CA125, glowing red in danger zones."
      >
        <div className="h-[400px]">
          {boundariesData ? (
            <ResponsiveContainer width="100%" height="100%">
              <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
                <XAxis
                  type="number"
                  dataKey="afp"
                  name="AFP"
                  domain={[0, 5000]}
                  stroke="#4b5563"
                  fontSize={10}
                  label={{
                    value: 'AFP (pg/ml)',
                    position: 'bottom',
                    fill: '#4b5563',
                    fontSize: 10
                  }}
                />
                <YAxis
                  type="number"
                  dataKey="ca125"
                  name="CA125"
                  domain={[0, 100]}
                  stroke="#4b5563"
                  fontSize={10}
                  label={{
                    value: 'CA125 (U/ml)',
                    angle: -90,
                    position: 'insideLeft',
                    fill: '#4b5563',
                    fontSize: 10
                  }}
                />
                <ZAxis type="number" dataKey="risk" range={[50, 400]} />
                <Tooltip
                  cursor={{ strokeDasharray: '3 3' }}
                  contentStyle={{ backgroundColor: '#0d1117', border: '1px solid #1f2937' }}
                  formatter={(value) => [`${value}%`, 'Risk Level']}
                />
                <Scatter data={boundariesData} fill="#ef4444">
                  {boundariesData.map((entry, index) => (
                    <Cell
                      key={`cell-${index}`}
                      fill={entry.risk > 50 ? '#ef4444' : '#10b981'}
                      opacity={entry.risk / 100}
                    />
                  ))}
                </Scatter>
              </ScatterChart>
            </ResponsiveContainer>
          ) : (
            <div className="w-full h-full flex flex-col items-center justify-center text-gray-500">
              Loading Topography...
            </div>
          )}
        </div>
      </AnalyticView>
      </div>
  )
}

export default Boundaries
