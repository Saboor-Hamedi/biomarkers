import React from 'react'
import AnalyticView from './AnalyticView'
import { ResponsiveContainer, XAxis, YAxis, CartesianGrid, Tooltip, Cell, BarChart, Bar, LabelList } from 'recharts'
import { Activity, BarChart as BarChartIcon } from 'lucide-react'


const Shap = ({ activeTab, shapData, shapTableData }) => {
  return (
<div className={activeTab === 'shap' ? 'block' : 'hidden'}>
        <AnalyticView
        title="SHAP Waterfall (Patient Logic)"
        icon={BarChartIcon}
        explanation="Deconstructs the mathematical journey of the current patient's prediction. Red bars push the risk higher; green bars pull it down."
        columns={['Feature', 'Impact', 'Input Value']}
        tableData={shapTableData}
      >
        <div className="h-[400px]">
          {shapData ? (
            <ResponsiveContainer width="100%" height="100%">
              <BarChart
                data={shapData}
                margin={{ top: 20, right: 30, left: 20, bottom: 5 }}
                layout="vertical"
              >
                <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" horizontal={false} />
                <XAxis
                  type="number"
                  stroke="#4b5563"
                  fontSize={10}
                  label={{
                    value: 'Risk Impact (%)',
                    position: 'bottom',
                    fill: '#4b5563',
                    fontSize: 10
                  }}
                />
                <YAxis
                  dataKey="feature"
                  type="category"
                  stroke="#4b5563"
                  fontSize={10}
                  width={100}
                />
                <Tooltip
                  contentStyle={{ backgroundColor: '#0d1117', border: '1px solid #1f2937' }}
                  formatter={(value) => [`${value > 0 ? '+' : ''}${value}%`, 'Impact']}
                />
                <Bar dataKey="value" radius={[0, 4, 4, 0]}>
                  {shapData.map((entry, index) => (
                    <Cell
                      key={`cell-${index}`}
                      fill={
                        entry.feature === 'Baseline'
                          ? '#3b82f6'
                          : entry.value > 0
                            ? '#ef4444'
                            : '#10b981'
                      }
                    />
                  ))}
                  <LabelList
                    dataKey="value"
                    position="right"
                    fill="#d1d5db"
                    fontSize={10}
                    fontWeight="bold"
                    formatter={(val, _name, props) =>
                      `${val > 0 && props?.payload?.feature !== 'Baseline' ? '+' : ''}${val}%`
                    }
                  />
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          ) : (
            <div className="w-full h-full flex flex-col items-center justify-center text-gray-500 opacity-50">
              <Activity size={32} className="mb-2" />
              <p className="text-[10px] font-bold tracking-widest">Awaiting Audit...</p>
            </div>
          )}
        </div>
      </AnalyticView>
      </div>
  )
}

export default Shap
