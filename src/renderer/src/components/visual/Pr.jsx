import React from 'react'
import AnalyticView from './AnalyticView'
import { ResponsiveContainer, XAxis, YAxis, CartesianGrid, Tooltip, LineChart, Line } from 'recharts'
import { Target, Zap } from 'lucide-react'


const Pr = ({ activeTab, prTableData, metrics }) => {
  return (
<div className={activeTab === 'pr' ? 'block' : 'hidden'}>
        <AnalyticView
        title="Precision-Recall"
        icon={Zap}
        explanation="Precision-Recall curves are critical for imbalanced clinical data. They show the trade-off between identifying true cases and avoiding false alarms."
        tableData={prTableData}
        columns={['Model', 'Avg Precision', 'Peak Recall']}
      >
        <div className="h-[400px]">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart margin={{ top: 5, right: 20, left: 0, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" vertical={false} />
              <XAxis dataKey="x" type="number" domain={[0, 1]} stroke="#4b5563" fontSize={10} />
              <YAxis domain={[0, 1]} stroke="#4b5563" fontSize={10} />
              <Tooltip contentStyle={{ backgroundColor: '#000', border: '1px solid #374151' }} />
              {metrics?.pr &&
                Object.entries(metrics.pr).map(([name, data]) => (
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
            </LineChart>
          </ResponsiveContainer>
        </div>
      </AnalyticView>
      </div>
  )
}

export default Pr
