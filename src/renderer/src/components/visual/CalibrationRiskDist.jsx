import React from 'react'
import AnalyticView from './AnalyticView'
import { ResponsiveContainer, XAxis, YAxis, CartesianGrid, Tooltip, Legend, LineChart, Line, BarChart, Bar } from 'recharts'
import { Target, BarChart as BarChartIcon } from 'lucide-react'


const CalibrationRiskDist = ({ activeTab, calibrationRiskData }) => {
  return (
<div className={activeTab === 'calibration-risk' ? 'block' : 'hidden'}>
      {(() => {
        const calData = calibrationRiskData
        const COLORS = {
          'Logistic Regression': '#8b5cf6',
          'Random Forest': '#10b981',
          Svm: '#f59e0b',
          Xgboost: '#ef4444',
          Logistic_Regression: '#8b5cf6',
          Random_Forest: '#10b981',
          SVM: '#f59e0b',
          XGBoost: '#ef4444'
        }
        const getColor = (name) => COLORS[name] || '#3b82f6'
        const strat = calData?.stratification
    const stratMax = strat ? Math.max(strat.safe, strat.moderate, strat.high, strat.critical) : 1
    const stratBars = strat
      ? [
          { label: 'Safe (<45%)', count: strat.safe, color: '#10b981' },
          { label: 'Moderate (45-60%)', count: strat.moderate, color: '#f59e0b' },
          { label: 'High (60-75%)', count: strat.high, color: '#f97316' },
          { label: 'Critical (>75%)', count: strat.critical, color: '#ef4444' }
        ]
      : []

    return (
      <AnalyticView
        title="Calibration & Risk Analysis"
        icon={Target}
        explanation="Four-panel forensic breakdown: model calibration reliability, best-model risk distribution, precision/recall/F1 threshold sweep, and cohort risk stratification counts."
      >
        <div className="grid grid-cols-1 xl:grid-cols-2 gap-8">
          {/* Panel 1: Model Calibration Curves */}
          <div className="space-y-3">
            <p className="text-[9px] font-black tracking-[0.2em] text-blue-500">
              Model Calibration Curves
            </p>
            <div className="h-[260px]">
              {calData?.calibration ? (
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart margin={{ top: 10, right: 10, bottom: 30, left: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
                    <XAxis
                      type="number"
                      dataKey="x"
                      domain={[0, 1]}
                      stroke="#4b5563"
                      fontSize={9}
                      label={{
                        value: 'Mean Predicted Probability',
                        position: 'bottom',
                        fill: '#6b7280',
                        fontSize: 9
                      }}
                    />
                    <YAxis
                      domain={[0, 1]}
                      stroke="#4b5563"
                      fontSize={9}
                      label={{
                        value: 'Fraction of Positives',
                        angle: -90,
                        position: 'insideLeft',
                        fill: '#6b7280',
                        fontSize: 9
                      }}
                    />
                    <Tooltip
                      contentStyle={{ backgroundColor: '#0d1117', border: '1px solid #1f2937' }}
                      formatter={(v) => [v.toFixed(3)]}
                    />
                    <Legend
                      iconType="circle"
                      wrapperStyle={{ fontSize: '9px', paddingTop: '8px' }}
                    />
                    {/* Perfect calibration diagonal */}
                    <Line
                      type="linear"
                      data={[
                        { x: 0, y: 0 },
                        { x: 1, y: 1 }
                      ]}
                      dataKey="y"
                      stroke="#374151"
                      strokeDasharray="5 5"
                      dot={false}
                      name="Perfect Calibration"
                    />
                    {Object.entries(calData.calibration).map(([name, points]) => (
                      <Line
                        key={name}
                        type="linear"
                        data={points}
                        dataKey="y"
                        stroke={getColor(name)}
                        dot={{ r: 4, fill: getColor(name) }}
                        strokeWidth={2}
                        name={name}
                      />
                    ))}
                  </LineChart>
                </ResponsiveContainer>
              ) : (
                <div className="w-full h-full flex items-center justify-center text-gray-600 text-[9px] font-bold">
                  Audit Required
                </div>
              )}
            </div>
          </div>

          {/* Panel 2: Risk Distribution */}
          <div className="space-y-3">
            <p className="text-[9px] font-black tracking-[0.2em] text-blue-500">
              {calData?.riskDistribution?.bestModel || 'Best Model'} — Risk Distribution
            </p>
            <div className="h-[260px]">
              {calData?.riskDistribution ? (
                (() => {
                  // Merge benign and malignant arrays into one dataset by x-bin
                  const merged = (calData.riskDistribution.benign || []).map((b, i) => ({
                    x: b.x,
                    benign: b.y,
                    malignant: calData.riskDistribution.malignant?.[i]?.y || 0
                  }))
                  return (
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart
                        data={merged}
                        margin={{ top: 10, right: 10, bottom: 30, left: 0 }}
                        barCategoryGap="1%"
                      >
                        <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" vertical={false} />
                        <XAxis
                          dataKey="x"
                          type="number"
                          domain={[0, 1]}
                          stroke="#4b5563"
                          fontSize={9}
                          tickFormatter={(v) => v.toFixed(1)}
                          label={{
                            value: 'Predicted Risk Probability',
                            position: 'bottom',
                            fill: '#6b7280',
                            fontSize: 9
                          }}
                        />
                        <YAxis
                          stroke="#4b5563"
                          fontSize={9}
                          label={{
                            value: 'Density',
                            angle: -90,
                            position: 'insideLeft',
                            fill: '#6b7280',
                            fontSize: 9
                          }}
                        />
                        <Tooltip
                          contentStyle={{
                            backgroundColor: '#0d1117',
                            border: '1px solid #1f2937'
                          }}
                        />
                        <Legend
                          iconType="square"
                          wrapperStyle={{ fontSize: '9px', paddingTop: '8px' }}
                        />
                        <Bar dataKey="benign" name="Benign" fill="#10b981" opacity={0.75} />
                        <Bar dataKey="malignant" name="Malignant" fill="#ef4444" opacity={0.75} />
                      </BarChart>
                    </ResponsiveContainer>
                  )
                })()
              ) : (
                <div className="w-full h-full flex items-center justify-center text-gray-600 text-[9px] font-bold">
                  Audit Required
                </div>
              )}
            </div>
          </div>

          {/* Panel 3: Threshold Optimization */}
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <p className="text-[9px] font-black tracking-[0.2em] text-blue-500">
                Threshold Optimization
              </p>
              {calData?.optimalThreshold !== undefined && (
                <span className="text-[8px] bg-purple-500/10 text-purple-400 border border-purple-500/20 px-2 py-0.5 rounded font-black">
                  Optimal: {calData.optimalThreshold}
                </span>
              )}
            </div>
            <div className="h-[260px]">
              {calData?.thresholdOptimization ? (
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart
                    data={calData.thresholdOptimization}
                    margin={{ top: 10, right: 10, bottom: 30, left: 0 }}
                  >
                    <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" vertical={false} />
                    <XAxis
                      dataKey="threshold"
                      type="number"
                      domain={[0, 1]}
                      stroke="#4b5563"
                      fontSize={9}
                      label={{
                        value: 'Classification Threshold',
                        position: 'bottom',
                        fill: '#6b7280',
                        fontSize: 9
                      }}
                    />
                    <YAxis
                      domain={[0, 1]}
                      stroke="#4b5563"
                      fontSize={9}
                      label={{
                        value: 'Score',
                        angle: -90,
                        position: 'insideLeft',
                        fill: '#6b7280',
                        fontSize: 9
                      }}
                    />
                    <Tooltip
                      contentStyle={{ backgroundColor: '#0d1117', border: '1px solid #1f2937' }}
                    />
                    <Legend
                      iconType="circle"
                      wrapperStyle={{ fontSize: '9px', paddingTop: '8px' }}
                    />
                    <Line
                      type="monotone"
                      dataKey="precision"
                      stroke="#3b82f6"
                      dot={false}
                      strokeWidth={2}
                      name="Precision"
                    />
                    <Line
                      type="monotone"
                      dataKey="recall"
                      stroke="#ef4444"
                      dot={false}
                      strokeWidth={2}
                      name="Recall"
                    />
                    <Line
                      type="monotone"
                      dataKey="f1"
                      stroke="#10b981"
                      dot={false}
                      strokeWidth={2}
                      name="F1-Score"
                    />
                  </LineChart>
                </ResponsiveContainer>
              ) : (
                <div className="w-full h-full flex items-center justify-center text-gray-600 text-[9px] font-bold">
                  Audit Required
                </div>
              )}
            </div>
          </div>

          {/* Panel 4: Risk Stratification Summary */}
          <div className="space-y-3">
            <p className="text-[9px] font-black tracking-[0.2em] text-blue-500">
              Risk Stratification Summary
            </p>
            <div className="h-[260px] flex flex-col justify-end gap-3 pb-8">
              {strat ? (
                stratBars.map((bar) => (
                  <div key={bar.label} className="space-y-1">
                    <div className="flex justify-between text-[8px] font-black tracking-widest">
                      <span style={{ color: bar.color }}>{bar.label}</span>
                      <span className="text-white font-mono">{bar.count}</span>
                    </div>
                    <div className="h-6 bg-gray-900 rounded overflow-hidden">
                      <div
                        className="h-full rounded transition-all duration-1000"
                        style={{
                          width: `${(bar.count / stratMax) * 100}%`,
                          backgroundColor: bar.color,
                          opacity: 0.85
                        }}
                      />
                    </div>
                  </div>
                ))
              ) : (
                <div className="w-full h-full flex items-center justify-center text-gray-600 text-[9px] font-bold">
                  Audit Required
                </div>
              )}
            </div>
          </div>
        </div>
      </AnalyticView>
        )
      })()}
      </div>
  )
}

export default CalibrationRiskDist
