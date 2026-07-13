import React from 'react'
import AnalyticView from './AnalyticView'
import { ResponsiveContainer, XAxis, YAxis, CartesianGrid, Tooltip, Legend, LineChart, Line } from 'recharts'
import { Activity } from 'lucide-react'


const Trajectory = ({ activeTab, trajectoryData, trajectoryModels, trajectoryColors, trajectoryTableData }) => {
  return (
<div className={activeTab === 'trajectory' ? 'block' : 'hidden'}>
        <AnalyticView
        title="Neural Risk Trajectories (PSA Sweep)"
        icon={Activity}
        explanation="This visualizes Partial Dependence Waves. We sweep the PSA biomarker from 0 to 20 while holding others constant, revealing the precise danger thresholds for each model in the ensemble."
        tableData={trajectoryTableData}
        columns={['Neural Model', '50% Risk Threshold', 'Boundary Status']}
      >
        <div className="h-[400px]">
          {trajectoryData && trajectoryData.length > 0 ? (
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={trajectoryData} margin={{ top: 20, right: 30, left: 0, bottom: 20 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" vertical={false} />
                <XAxis
                  dataKey="psa"
                  type="number"
                  domain={[0, 20]}
                  stroke="#4b5563"
                  fontSize={10}
                  label={{
                    value: 'PSA Level (pg/ml)',
                    position: 'bottom',
                    fill: '#4b5563',
                    fontSize: 10
                  }}
                />
                <YAxis
                  domain={[0, 100]}
                  stroke="#4b5563"
                  fontSize={10}
                  label={{
                    value: 'Risk Probability (%)',
                    angle: -90,
                    position: 'insideLeft',
                    fill: '#4b5563',
                    fontSize: 10
                  }}
                />
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#0d1117',
                    border: '1px solid #1f2937',
                    borderRadius: '8px'
                  }}
                  itemStyle={{ fontSize: '10px', fontWeight: 'bold' }}
                  labelStyle={{ color: '#9ca3af', fontSize: '10px', paddingBottom: '8px' }}
                  formatter={(value) => [`${value}%`, undefined]}
                  labelFormatter={(label) => `PSA Level: ${label} pg/ml`}
                />
                <Legend iconType="circle" wrapperStyle={{ fontSize: '10px', paddingTop: '20px' }} />
                {trajectoryModels.map((modelName, index) => (
                  <Line
                    key={modelName}
                    type="monotone"
                    dataKey={modelName}
                    stroke={trajectoryColors[index % trajectoryColors.length]}
                    strokeWidth={3}
                    dot={false}
                    activeDot={{
                      r: 6,
                      fill: trajectoryColors[index % trajectoryColors.length],
                      stroke: '#000',
                      strokeWidth: 2
                    }}
                    animationDuration={2000}
                  />
                ))}
              </LineChart>
            </ResponsiveContainer>
          ) : (
            <div className="w-full h-full flex flex-col items-center justify-center text-gray-500 opacity-50">
              <Activity size={32} className="mb-2" />
              <p className="text-[10px] font-bold tracking-widest">
                Generating Trajectories...
              </p>
              <p className="text-[8px] tracking-wider mt-2">
                Requires patient input prediction
              </p>
            </div>
          )}
        </div>
      </AnalyticView>
      </div>
  )
}

export default Trajectory
