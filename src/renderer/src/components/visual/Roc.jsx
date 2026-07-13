import React from 'react'
import AnalyticView from './AnalyticView'
import { ResponsiveContainer, XAxis, YAxis, CartesianGrid, Tooltip, LineChart, Line } from 'recharts'
import { Target } from 'lucide-react'


const Roc = ({ activeTab, rocTableData, metrics }) => {
  return (
<div className={activeTab === 'roc' ? 'block' : 'hidden'}>
        <AnalyticView
        title="ROC Performance"
        icon={Target}
        explanation="The ROC Curve (Receiver Operating Characteristic) measures model discrimination ability. Higher curves toward the top-left indicate superior sensitivity and specificity."
        tableData={rocTableData}
        columns={['Model', 'AUC Score', 'Confidence']}
      >
        <div className="h-[400px]">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart margin={{ top: 5, right: 20, left: 0, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" vertical={false} />
              <XAxis dataKey="x" type="number" domain={[0, 1]} stroke="#4b5563" fontSize={10} />
              <YAxis domain={[0, 1]} stroke="#4b5563" fontSize={10} />
              <Tooltip contentStyle={{ backgroundColor: '#000', border: '1px solid #374151' }} />
              {metrics?.roc &&
                Object.entries(metrics.roc).map(([name, data]) => (
                  <Line
                    key={name}
                    type="monotone"
                    data={data.points}
                    dataKey="y"
                    stroke={data.color}
                    dot={false}
                    strokeWidth={3}
                  />
                ))}
              <Line
                type="monotone"
                data={[
                  { x: 0, y: 0 },
                  { x: 1, y: 1 }
                ]}
                dataKey="y"
                stroke="#374151"
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

export default Roc
