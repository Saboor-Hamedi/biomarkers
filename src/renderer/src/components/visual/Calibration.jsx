import React from 'react'
import AnalyticView from './AnalyticView'
import { ResponsiveContainer, XAxis, YAxis, CartesianGrid, Tooltip, LineChart, Line } from 'recharts'
import { Target } from 'lucide-react'


const Calibration = ({ activeTab, calibrationTableData, metrics }) => {
  return (
<div className={activeTab === 'calibration' ? 'block' : 'hidden'}>
        <AnalyticView
        title="Model Calibration (Reliability Diagram)"
        icon={Target}
        explanation="Calibration curves show how closely the predicted probabilities align with the true fraction of positive cases. A perfectly calibrated model follows the diagonal line."
        tableData={calibrationTableData}
        columns={['Model', 'Brier Score', 'Status']}
      >
        <div className="h-[400px]">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart margin={{ top: 5, right: 20, left: 0, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" vertical={false} />
              <XAxis
                dataKey="predicted"
                type="number"
                domain={[0, 1]}
                stroke="#4b5563"
                fontSize={10}
                label={{
                  value: 'Mean Predicted Probability',
                  position: 'bottom',
                  fill: '#4b5563',
                  fontSize: 10
                }}
              />
              <YAxis
                dataKey="true_fraction"
                domain={[0, 1]}
                stroke="#4b5563"
                fontSize={10}
                label={{
                  value: 'Fraction of Positives',
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
              />
              {metrics?.calibration &&
                Object.entries(metrics.calibration).map(([name, data]) => (
                  <Line
                    key={name}
                    type="monotone"
                    data={data.points}
                    dataKey="true_fraction"
                    stroke={data.color}
                    dot={{ r: 3, fill: data.color }}
                    strokeWidth={2}
                  />
                ))}
              {/* Perfectly Calibrated Diagonal Line */}
              <Line
                type="monotone"
                data={[
                  { predicted: 0, true_fraction: 0 },
                  { predicted: 1, true_fraction: 1 }
                ]}
                dataKey="true_fraction"
                stroke="#6b7280"
                strokeDasharray="5 5"
                dot={false}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </AnalyticView>
      </div>
  )
}

export default Calibration
