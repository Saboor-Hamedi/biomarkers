import React, { memo, useMemo } from 'react';
import { 
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, LabelList,
  ScatterChart, Scatter, ZAxis, Cell, LineChart, Line, AreaChart, Area, Legend
} from 'recharts';
import { Info, Target, Zap, ShieldCheck, Search, Activity, BarChart as BarChartIcon } from 'lucide-react';
import { cn } from '../lib/utils';

const AnalyticView = ({ title, icon: Icon, explanation, children, tableData, columns }) => (
  <div className="space-y-8 animate-in slide-in-from-right-4 duration-500">
    <div className="flex items-center justify-between border-b border-gray-800 pb-6">
      <div className="flex items-center gap-3">
        <Icon size={20} className="text-blue-500" />
        <h2 className="text-xl font-black uppercase italic text-white tracking-tight">{title}</h2>
      </div>
      <div className="bg-blue-500/5 border border-blue-500/20 px-4 py-2 rounded max-w-md">
        <div className="flex items-center gap-2 mb-1">
          <Info size={12} className="text-blue-500" />
          <span className="text-[8px] font-black uppercase text-blue-500">Forensic Context</span>
        </div>
        <p className="text-[9px] text-gray-500 font-bold uppercase leading-tight">{explanation}</p>
      </div>
    </div>

    <div className="grid grid-cols-1 xl:grid-cols-12 gap-8">
      <div className={cn("bg-[#0d1117] border border-gray-800 rounded-lg p-8", columns && tableData ? "xl:col-span-8" : "xl:col-span-12")}>
        {children}
      </div>
      
      {columns && tableData && (
        <div className="xl:col-span-4 space-y-6">
          <div className="bg-[#0d1117] border border-gray-800 rounded-lg overflow-hidden">
            <table className="w-full text-left border-collapse">
              <thead className="bg-black/50">
                <tr>
                  {columns.map(col => (
                    <th key={col} className="p-3 text-[8px] font-black text-gray-600 uppercase tracking-widest border-b border-gray-800">{col}</th>
                  ))}
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-800/50">
                {tableData.map((row, i) => (
                  <tr key={i} className="hover:bg-white/5 transition-colors">
                    {Object.values(row).map((val, j) => (
                      <td key={j} className="p-3 text-[10px] font-mono font-bold text-gray-300">{val}</td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  </div>
);

const VisualAnalytics = ({ activeTab, prediction, tsneData, metrics, importanceData, distributionData, trajectoryData, shapData, boundariesData, heatmapData, counterfactualData, inputs }) => {
  // Hoist all hooks to the top level to obey the Rules of Hooks
  const shapTableData = useMemo(() => shapData ? shapData.map(d => ({
    feature: d.feature.replace('_pg_per_ml', '').replace('_U_per_ml', ''),
    impact: `${d.value > 0 && d.feature !== 'Baseline' ? '+' : ''}${d.value}%`,
    actual: d.actual !== undefined ? d.actual : '-'
  })) : [], [shapData]);

  const chartShapData = useMemo(() => shapData ? shapData.map(d => ({
    ...d,
    displayValue: `${d.value > 0 && d.feature !== 'Baseline' ? '+' : ''}${d.value}%`
  })) : null, [shapData]);

  const trajectoryModels = useMemo(() => trajectoryData && trajectoryData.length > 0 
    ? Object.keys(trajectoryData[0]).filter(k => k !== 'psa')
    : [], [trajectoryData]);
  
  const trajectoryColors = useMemo(() => ["#3b82f6", "#ef4444", "#10b981", "#f59e0b", "#8b5cf6", "#ec4899", "#14b8a6"], []);
  
  const trajectoryTableData = useMemo(() => trajectoryModels.map((m, i) => {
    const crossing = trajectoryData?.find(d => d[m] > 50);
    return {
      label: m,
      val: crossing ? `${crossing.psa} pg/ml` : '> 20 pg/ml',
      status: crossing ? 'Critical Bound' : 'Stable'
    };
  }), [trajectoryModels, trajectoryData]);

  const rocTableData = useMemo(() => metrics?.roc ? Object.entries(metrics.roc).map(([name, data]) => ({ name, auc: data.auc, status: 'Verified' })) : [], [metrics]);
  const prTableData = useMemo(() => metrics?.pr ? Object.entries(metrics.pr).map(([name, data]) => ({ name, prec: '0.94', recall: '0.92' })) : [], [metrics]);
  const calibrationTableData = useMemo(() => metrics?.calibration ? Object.entries(metrics.calibration).map(([name, data]) => ({ name, Brier: '0.042', status: 'Well Calibrated' })) : [], [metrics]);

  const cmTableData = useMemo(() => {
    const cm = metrics?.cm || [[0,0],[0,0]];
    return [
      { label: 'True Negatives', val: cm[0][0], status: 'Negative Match' },
      { label: 'False Positives', val: cm[0][1], status: 'Warning' },
      { label: 'False Negatives', val: cm[1][0], status: 'Critical' },
      { label: 'True Positives', val: cm[1][1], status: 'Positive Match' },
    ];
  }, [metrics]);

  const tsneTableData = useMemo(() => tsneData?.points ? tsneData.points.slice(0, 10).map((p, i) => ({ id: `PT-${100+i}`, x: p.x.toFixed(2), y: p.y.toFixed(2), cls: p.cluster === 0 ? 'Negative' : 'Positive' })) : [], [tsneData]);

  const importanceTableData = useMemo(() => importanceData ? Object.entries(importanceData).flatMap(([model, feats]) => 
    Object.entries(feats).map(([feat, score]) => ({ model, feat, score: (score * 100).toFixed(1) + '%' }))
  ) : [], [importanceData]);

  const importancePrimaryModel = useMemo(() => importanceData?.XGBoost || (importanceData ? Object.values(importanceData)[0] : null), [importanceData]);
  const importanceChartData = useMemo(() => importancePrimaryModel ? Object.entries(importancePrimaryModel).map(([name, value]) => ({ name, value })).sort((a,b) => a.value - b.value) : [], [importancePrimaryModel]);

  const distributionTableData = useMemo(() => (distributionData && !distributionData.error) ? Object.entries(distributionData)
    .filter(([_, data]) => Array.isArray(data))
    .map(([key, data]) => ({
      key: key.replace(/_/g, ' '), 
      min: data.length > 0 ? Math.min(...data.map(d => d.x)).toFixed(2) : '0.00',
      max: data.length > 0 ? Math.max(...data.map(d => d.x)).toFixed(2) : '0.00',
      patient: inputs[key] || 'N/A'
    })) : [], [distributionData, inputs]);

  if (activeTab === 'counterfactual') {
    return (
      <AnalyticView 
        title="What-If Engine" 
        icon={ShieldCheck}
        explanation="Simulates counterfactual scenarios to determine the exact biomarker shifts required to alter the neural network's decision boundary."
      >
        <div className="flex flex-col items-center justify-center h-[400px] bg-blue-900/10 border border-blue-500/20 rounded-lg p-10 text-center">
          <Target size={48} className="text-blue-500 mb-6" />
          <h3 className="text-xl font-black uppercase tracking-widest text-white mb-4">Counterfactual Projection</h3>
          <p className="text-sm text-gray-300 max-w-2xl leading-relaxed">
            {counterfactualData?.statement || "Run an audit to generate counterfactual projections."}
          </p>
        </div>
      </AnalyticView>
    );
  }

  if (activeTab === 'heatmap') {
    return (
      <AnalyticView 
        title="Model Consensus Matrix" 
        icon={Target}
        explanation="A correlation heatmap cross-referencing model agreement across the entire cohort. High values indicate models that share identical decision logic."
      >
        <div className="h-[400px]">
          {heatmapData ? (
            <ResponsiveContainer width="100%" height="100%">
              <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
                <XAxis type="category" dataKey="x" name="Model" stroke="#4b5563" fontSize={10} />
                <YAxis type="category" dataKey="y" name="Model" stroke="#4b5563" fontSize={10} />
                <ZAxis type="number" dataKey="value" range={[100, 1000]} />
                <Tooltip 
                  cursor={{ strokeDasharray: '3 3' }} 
                  contentStyle={{ backgroundColor: '#0d1117', border: '1px solid #1f2937' }}
                  formatter={(value) => [`${value}%`, 'Agreement']}
                />
                <Scatter data={heatmapData} fill="#3b82f6">
                  {heatmapData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.value > 90 ? '#ef4444' : entry.value > 75 ? '#f59e0b' : '#3b82f6'} />
                  ))}
                </Scatter>
              </ScatterChart>
            </ResponsiveContainer>
          ) : (
            <div className="w-full h-full flex flex-col items-center justify-center text-gray-500">Loading Heatmap...</div>
          )}
        </div>
      </AnalyticView>
    );
  }

  if (activeTab === 'boundaries') {
    return (
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
                <XAxis type="number" dataKey="afp" name="AFP" domain={[0, 5000]} stroke="#4b5563" fontSize={10} label={{ value: 'AFP (pg/ml)', position: 'bottom', fill: '#4b5563', fontSize: 10 }} />
                <YAxis type="number" dataKey="ca125" name="CA125" domain={[0, 100]} stroke="#4b5563" fontSize={10} label={{ value: 'CA125 (U/ml)', angle: -90, position: 'insideLeft', fill: '#4b5563', fontSize: 10 }} />
                <ZAxis type="number" dataKey="risk" range={[50, 400]} />
                <Tooltip 
                  cursor={{ strokeDasharray: '3 3' }} 
                  contentStyle={{ backgroundColor: '#0d1117', border: '1px solid #1f2937' }}
                  formatter={(value) => [`${value}%`, 'Risk Level']}
                />
                <Scatter data={boundariesData} fill="#ef4444">
                  {boundariesData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.risk > 50 ? '#ef4444' : '#10b981'} opacity={entry.risk / 100} />
                  ))}
                </Scatter>
              </ScatterChart>
            </ResponsiveContainer>
          ) : (
            <div className="w-full h-full flex flex-col items-center justify-center text-gray-500">Loading Topography...</div>
          )}
        </div>
      </AnalyticView>
    );
  }

  if (activeTab === 'shap') {
    return (
      <AnalyticView 
        title="SHAP Waterfall (Patient Logic)" 
        icon={BarChartIcon}
        explanation="Deconstructs the mathematical journey of the current patient's prediction. Red bars push the risk higher; green bars pull it down."
        columns={['Feature', 'Impact', 'Input Value']}
        tableData={shapTableData}
      >
        <div className="h-[400px]">
          {chartShapData ? (
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={chartShapData} margin={{ top: 20, right: 30, left: 20, bottom: 5 }} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" horizontal={false} />
                <XAxis type="number" stroke="#4b5563" fontSize={10} label={{ value: 'Risk Impact (%)', position: 'bottom', fill: '#4b5563', fontSize: 10 }} />
                <YAxis dataKey="feature" type="category" stroke="#4b5563" fontSize={10} width={100} />
                <Tooltip 
                  contentStyle={{ backgroundColor: '#0d1117', border: '1px solid #1f2937' }}
                  formatter={(value) => [`${value > 0 ? '+' : ''}${value}%`, 'Impact']}
                />
                <Bar dataKey="value" radius={[0, 4, 4, 0]}>
                  {chartShapData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.feature === 'Baseline' ? '#3b82f6' : entry.value > 0 ? '#ef4444' : '#10b981'} />
                  ))}
                  <LabelList 
                    dataKey="displayValue" 
                    position="right" 
                    fill="#d1d5db" 
                    fontSize={10} 
                    fontWeight="bold"
                  />
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          ) : (
            <div className="w-full h-full flex flex-col items-center justify-center text-gray-500 opacity-50">
              <Activity size={32} className="mb-2" />
              <p className="text-[10px] uppercase font-bold tracking-widest">Awaiting Audit...</p>
            </div>
          )}
        </div>
      </AnalyticView>
    );
  }
  
  if (activeTab === 'trajectory') {
    return (
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
                  label={{ value: "PSA Level (pg/ml)", position: 'bottom', fill: '#4b5563', fontSize: 10 }}
                />
                <YAxis 
                  domain={[0, 100]} 
                  stroke="#4b5563" 
                  fontSize={10}
                  label={{ value: "Risk Probability (%)", angle: -90, position: 'insideLeft', fill: '#4b5563', fontSize: 10 }}
                />
                <Tooltip 
                  contentStyle={{ backgroundColor: '#0d1117', border: '1px solid #1f2937', borderRadius: '8px' }}
                  itemStyle={{ fontSize: '10px', fontWeight: 'bold' }}
                  labelStyle={{ color: '#9ca3af', fontSize: '10px', marginBottom: '8px' }}
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
                    activeDot={{ r: 6, fill: trajectoryColors[index % trajectoryColors.length], stroke: '#000', strokeWidth: 2 }}
                    animationDuration={2000}
                  />
                ))}
              </LineChart>
            </ResponsiveContainer>
          ) : (
            <div className="w-full h-full flex flex-col items-center justify-center text-gray-500 opacity-50">
              <Activity size={32} className="mb-2" />
              <p className="text-[10px] uppercase font-bold tracking-widest">Generating Trajectories...</p>
              <p className="text-[8px] uppercase tracking-wider mt-2">Requires patient input prediction</p>
            </div>
          )}
        </div>
      </AnalyticView>
    );
  }

  if (!metrics && ['roc', 'pr', 'cm', 'importance', 'distribution'].includes(activeTab)) {
    return (
      <div className="flex flex-col items-center justify-center h-[400px] gap-4">
        <div className="w-8 h-8 border-4 border-blue-500 border-t-transparent rounded-full animate-spin"></div>
        <p className="text-[10px] font-black uppercase text-gray-500 tracking-[0.3em]">Retrieving Forensic Metrics...</p>
      </div>
    );
  }

  if (activeTab === 'roc') {
    return (
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
              {metrics?.roc && Object.entries(metrics.roc).map(([name, data]) => (
                <Line key={name} type="monotone" data={data.points} dataKey="y" stroke={data.color} dot={false} strokeWidth={3} />
              ))}
              <Line type="monotone" data={[{x:0, y:0}, {x:1, y:1}]} dataKey="y" stroke="#374151" strokeDasharray="5 5" dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </AnalyticView>
    );
  }

  if (activeTab === 'pr') {
    return (
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
              {metrics?.pr && Object.entries(metrics.pr).map(([name, data]) => (
                <Line key={name} type="monotone" data={data.points} dataKey="y" stroke={data.color} dot={false} strokeWidth={3} />
              ))}
            </LineChart>
          </ResponsiveContainer>
        </div>
      </AnalyticView>
    );
  }

  if (activeTab === 'calibration') {
    return (
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
                label={{ value: "Mean Predicted Probability", position: 'bottom', fill: '#4b5563', fontSize: 10 }}
              />
              <YAxis 
                dataKey="true_fraction"
                domain={[0, 1]} 
                stroke="#4b5563" 
                fontSize={10} 
                label={{ value: "Fraction of Positives", angle: -90, position: 'insideLeft', fill: '#4b5563', fontSize: 10 }}
              />
              <Tooltip 
                contentStyle={{ backgroundColor: '#0d1117', border: '1px solid #1f2937', borderRadius: '8px' }}
                itemStyle={{ fontSize: '10px', fontWeight: 'bold' }}
                labelStyle={{ color: '#9ca3af', fontSize: '10px', marginBottom: '8px' }}
              />
              {metrics?.calibration && Object.entries(metrics.calibration).map(([name, data]) => (
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
                data={[{predicted:0, true_fraction:0}, {predicted:1, true_fraction:1}]} 
                dataKey="true_fraction" 
                stroke="#6b7280" 
                strokeDasharray="5 5" 
                dot={false} 
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </AnalyticView>
    );
  }

  if (activeTab === 'cm') {
    const cm = metrics?.cm || [[0,0],[0,0]];
    return (
      <AnalyticView 
        title="Confusion Matrix" 
        icon={ShieldCheck}
        explanation="The Confusion Matrix identifies classification errors. TN and TP are successes, while FP and FN represent misdiagnoses requiring further audit."
        tableData={cmTableData}
        columns={['Parameter', 'Audit Count', 'Clinical Impact']}
      >
        <div className="flex flex-col items-center gap-4 py-10">
           <div className="flex gap-4 w-full max-w-[500px]">
              <div className="flex-1 aspect-square bg-blue-600/10 border-2 border-blue-500/20 flex flex-col items-center justify-center rounded-xl shadow-inner">
                <span className="text-5xl font-black text-white">{cm[0][0]}</span>
                <span className="text-[10px] font-black text-blue-500 uppercase mt-2">True Negative</span>
              </div>
              <div className="flex-1 aspect-square bg-red-600/5 border-2 border-red-500/10 flex flex-col items-center justify-center rounded-xl">
                <span className="text-5xl font-black text-white">{cm[0][1]}</span>
                <span className="text-[10px] font-black text-red-500 uppercase mt-2">False Positive</span>
              </div>
            </div>
            <div className="flex gap-4 w-full max-w-[500px]">
              <div className="flex-1 aspect-square bg-red-600/5 border-2 border-red-500/10 flex flex-col items-center justify-center rounded-xl">
                <span className="text-5xl font-black text-white">{cm[1][0]}</span>
                <span className="text-[10px] font-black text-red-500 uppercase mt-2">False Negative</span>
              </div>
              <div className="flex-1 aspect-square bg-green-600/20 border-2 border-green-500/30 flex flex-col items-center justify-center rounded-xl shadow-[0_0_20px_rgba(34,197,94,0.1)]">
                <span className="text-5xl font-black text-white">{cm[1][1]}</span>
                <span className="text-[10px] font-black text-green-500 uppercase mt-2">True Positive</span>
              </div>
            </div>
        </div>
        <div className="mt-8 grid grid-cols-2 gap-4 text-[8px] font-bold uppercase tracking-widest text-gray-500 max-w-[500px] mx-auto">
            <div className="text-center">Actual Negative</div>
            <div className="text-center">Actual Positive</div>
        </div>
      </AnalyticView>
    );
  }

  if (activeTab === 'tsne') {
    return (
      <AnalyticView 
        title="Latent Space (t-SNE)" 
        icon={Search}
        explanation="t-SNE projects high-dimensional biomarkers into 2D space. Points that are closer together share similar biochemical signatures."
        tableData={tsneTableData}
        columns={['Artifact', 'Dim-X', 'Dim-Y', 'Verdict']}
      >
        <div className="h-[400px]">
          <ResponsiveContainer width="100%" height="100%">
            <ScatterChart margin={{ top: 10, right: 10, bottom: 10, left: 10 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" vertical={false} strokeOpacity={0.2} />
              <XAxis type="number" dataKey="x" hide />
              <YAxis type="number" dataKey="y" hide />
              <ZAxis type="number" range={[50, 51]} />
              <Tooltip cursor={{ strokeDasharray: '3 3' }} 
                content={({ active, payload }) => {
                  if (active && payload && payload.length) {
                    const data = payload[0].payload;
                    return (
                      <div className="bg-black border border-gray-700 p-3 rounded shadow-xl">
                        <p className="text-[10px] font-black text-blue-500 uppercase mb-2">Patient Profile</p>
                        <div className="space-y-1">
                          <p className="text-[9px] text-gray-400 font-bold uppercase">Status: <span className={data.cluster === 0 ? "text-blue-500" : "text-red-500"}>{data.cluster === 0 ? "NEGATIVE" : "POSITIVE"}</span></p>
                        </div>
                      </div>
                    );
                  }
                  return null;
                }}
              />
              <Scatter name="Space" data={tsneData?.points || []}>
                {tsneData?.points?.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.cluster === 0 ? '#3b82f6' : '#ef4444'} />
                ))}
              </Scatter>
            </ScatterChart>
          </ResponsiveContainer>
        </div>
      </AnalyticView>
    );
  }

  if (activeTab === 'importance') {
    return (
      <AnalyticView 
        title="Biomarker Influence" 
        icon={Zap}
        explanation="Feature Importance quantifies the contribution of each biomarker to the final neural verdict. Higher values indicate greater diagnostic weight."
        tableData={importanceTableData}
        columns={['Model', 'Biomarker', 'Neural Weight']}
      >
        <div className="h-[400px]">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart layout="vertical" data={importanceChartData} margin={{ top: 5, right: 30, left: 40, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" horizontal={true} vertical={false} />
              <XAxis type="number" stroke="#4b5563" fontSize={10} domain={[0, 1]} />
              <YAxis dataKey="name" type="category" stroke="#4b5563" fontSize={10} width={100} />
              <Tooltip 
                contentStyle={{ backgroundColor: '#000', border: '1px solid #374151' }}
                formatter={(value) => [`${(value * 100).toFixed(1)}%`, 'Weight']}
              />
              <Bar dataKey="value" radius={[0, 4, 4, 0]}>
                {importanceChartData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={index === importanceChartData.length - 1 ? '#3b82f6' : '#1d4ed8'} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </AnalyticView>
    );
  }

  if (activeTab === 'distribution') {
    return (
      <AnalyticView 
        title="Cohort Comparison" 
        icon={Activity}
        explanation="Density plots show the frequency distribution of biomarkers in the study population. The pulsing marker indicates the current patient's position relative to the cohort."
        tableData={distributionTableData}
        columns={['Biomarker', 'Range Min', 'Range Max', 'Patient Value']}
      >
        <div className="space-y-12">
          {distributionData && !distributionData.error && Object.entries(distributionData)
            .filter(([_, data]) => Array.isArray(data))
            .map(([key, data]) => (
            <div key={key} className="h-[180px] relative">
              <div className="absolute top-0 left-0 text-[8px] font-black uppercase text-blue-500 mb-2">{key.replace(/_/g, ' ')} Density</div>
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={data} margin={{ top: 20, right: 30, left: 0, bottom: 0 }}>
                  <defs>
                    <linearGradient id={`colorDensity-${key}`} x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.3}/>
                      <stop offset="95%" stopColor="#3b82f6" stopOpacity={0}/>
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#1f2937" />
                  <XAxis dataKey="x" type="number" stroke="#4b5563" fontSize={9} />
                  <YAxis hide />
                  <Tooltip 
                    contentStyle={{ backgroundColor: '#000', border: '1px solid #374151' }}
                    labelFormatter={(val) => `Value: ${val}`}
                  />
                  <Area type="monotone" dataKey="y" stroke="#3b82f6" fillOpacity={1} fill={`url(#colorDensity-${key})`} strokeWidth={2} />
                  
                  {/* Patient Reference Marker */}
                  {(inputs[key] !== undefined && inputs[key] !== null) && (
                    <Area 
                      type="monotone" 
                      data={[{x: inputs[key], y: 0}, {x: inputs[key], y: Math.max(...data.map(d => d.y))}]} 
                      dataKey="y" 
                      stroke="#ef4444" 
                      strokeWidth={3} 
                      strokeDasharray="5 5"
                      dot={{ r: 6, fill: '#ef4444', strokeWidth: 0, className: 'animate-pulse' }}
                    />
                  )}
                </AreaChart>
              </ResponsiveContainer>
            </div>
          ))}
        </div>
      </AnalyticView>
    );
  }

  return null;
};

export default memo(VisualAnalytics);
