import React from 'react'
import AnalyticView from './AnalyticView'
import { ResponsiveContainer, ScatterChart, Scatter, XAxis, YAxis, ZAxis, CartesianGrid, Tooltip, Cell, Legend } from 'recharts'
import { Maximize2, LayoutGrid, Search } from 'lucide-react'
import { cn } from '../../lib/utils'


const Tsne = ({ activeTab, tsneData, tsneSubsets, tsneView, setTsneView, tsneTableData }) => {
  return (
<div className={activeTab === 'tsne' ? 'block' : 'hidden'}>
      {(() => {
        const getProbColor = (p) => {
          if (p < 0.25) return `rgb(68, 1, 84)`
          if (p < 0.5) return `rgb(49, 104, 142)`
          if (p < 0.75) return `rgb(53, 183, 121)`
          return `rgb(253, 231, 37)`
        }

        return (
          <AnalyticView
        title="Latent Space Diagnostics"
        icon={Search}
        explanation="Multi-dimensional projection of high-fidelity biomarker signatures."
        tableData={tsneTableData}
        columns={['Sample', 't-SNE X', 't-SNE Y', 'Verdict', 'Risk %']}
        extraAction={
          <div className="flex bg-black/40 backdrop-blur-md rounded-full p-1 border border-white/5">
            <button
              onClick={() => setTsneView('standard')}
              className={cn(
                'px-4 py-1.5 text-[7px] font-black rounded-full transition-all flex items-center gap-2',
                tsneView === 'standard'
                  ? 'bg-blue-500 text-white shadow-[0_0_15px_rgba(59,130,246,0.5)]'
                  : 'text-gray-500 hover:text-gray-300'
              )}
            >
              <Maximize2 size={10} />
              Core View
            </button>
            <button
              onClick={() => setTsneView('audit')}
              className={cn(
                'px-4 py-1.5 text-[7px] font-black rounded-full transition-all flex items-center gap-2',
                tsneView === 'audit'
                  ? 'bg-purple-500 text-white shadow-[0_0_15px_rgba(168,85,247,0.5)]'
                  : 'text-gray-500 hover:text-gray-300'
              )}
            >
              <LayoutGrid size={10} />
              Neural Audit
            </button>
          </div>
        }
      >
        {tsneView === 'standard' ? (
          <div className="h-[550px] bg-black/20 rounded-xl border border-white/5 p-4">
            <ResponsiveContainer width="100%" height="100%">
              <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
                <CartesianGrid
                  strokeDasharray="3 3"
                  stroke="#1f2937"
                  vertical={false}
                  strokeOpacity={0.1}
                />
                <XAxis type="number" dataKey="x" hide />
                <YAxis type="number" dataKey="y" hide />
                <ZAxis type="number" range={[25, 26]} />
                <Tooltip
                  cursor={{ strokeDasharray: '3 3' }}
                  content={({ active, payload }) => {
                    if (active && payload && payload.length) {
                      const data = payload[0].payload
                      return (
                        <div className="bg-black/90 backdrop-blur-xl border border-white/10 p-4 rounded-lg shadow-2xl">
                          <p className="text-[10px] font-black text-blue-400 mb-3 tracking-widest">
                            Diagnostic Profile
                          </p>
                          <div className="space-y-2">
                            <div className="flex justify-between gap-8">
                              <span className="text-[8px] text-gray-500 font-bold">
                                Sample ID
                              </span>
                              <span className="text-[8px] text-white font-mono">
                                {data.sample_id}
                              </span>
                            </div>
                            <div className="flex justify-between gap-8">
                              <span className="text-[8px] text-gray-500 font-bold">
                                Clinical
                              </span>
                              <span
                                className={cn(
                                  'text-[8px] font-black',
                                  data.true_label === 0 ? 'text-blue-500' : 'text-red-500'
                                )}
                              >
                                {data.true_label === 0 ? 'BENIGN' : 'MALIGNANT'}
                              </span>
                            </div>
                            <div className="flex justify-between gap-8 border-t border-white/5 pt-2">
                              <span className="text-[8px] text-gray-500 font-bold">
                                Risk Score
                              </span>
                              <span className="text-[8px] text-white font-black">
                                {(data.probability * 100).toFixed(1)}%
                              </span>
                            </div>
                          </div>
                        </div>
                      )
                    }
                    return null
                  }}
                />
                <Legend
                  wrapperStyle={{
                    fontSize: '8px',
                    paddingTop: '30px',
                    textTransform: 'uppercase',
                    fontWeight: '900',
                    letterSpacing: '0.1em'
                  }}
                />
                <Scatter
                  name="Benign Signature"
                  data={tsneSubsets.benign}
                  fill="#3b82f6"
                  fillOpacity={0.8}
                  isAnimationActive={false}
                />
                <Scatter
                  name="Malignant Signature"
                  data={tsneSubsets.malignant}
                  fill="#ef4444"
                  fillOpacity={0.8}
                  shape="triangle"
                  isAnimationActive={false}
                />
              </ScatterChart>
            </ResponsiveContainer>
          </div>
        ) : (
          <div className="space-y-12">
            {/* Plot 1: t-SNE by True Label */}
            <div className="bg-black/30 p-6 rounded-2xl border border-white/5 relative">
              <div className="flex items-center justify-between mb-6">
                <div className="text-[8px] font-black text-blue-500 tracking-widest">
                  t-SNE: Clinical Ground Truth
                </div>
                <div className="h-px flex-1 bg-gradient-to-r from-blue-500/20 to-transparent ml-4" />
              </div>
              <div className="h-[350px]">
                <ResponsiveContainer width="100%" height="100%">
                  <ScatterChart margin={{ top: 10, right: 10, bottom: 10, left: 10 }}>
                    <CartesianGrid
                      strokeDasharray="3 3"
                      stroke="#1f2937"
                      vertical={false}
                      strokeOpacity={0.1}
                    />
                    <XAxis type="number" dataKey="x" hide />
                    <YAxis type="number" dataKey="y" hide />
                    <ZAxis type="number" range={[15, 16]} />
                    <Tooltip
                      cursor={{ strokeDasharray: '3 3' }}
                      content={({ active, payload }) => {
                        if (active && payload && payload.length) {
                          const data = payload[0].payload
                          return (
                            <div className="bg-black/90 border border-white/10 p-2 rounded shadow-xl">
                              <p className="text-[7px] font-black text-blue-500 tracking-tighter">
                                Sample: {data.sample_id}
                              </p>
                            </div>
                          )
                        }
                        return null
                      }}
                    />
                    <Scatter data={tsneData?.points || []} isAnimationActive={false}>
                      {tsneData?.points?.map((entry, index) => (
                        <Cell
                          key={`cell-${index}`}
                          fill={entry.true_label === 0 ? '#3b82f6' : '#ef4444'}
                          fillOpacity={0.7}
                        />
                      ))}
                    </Scatter>
                  </ScatterChart>
                </ResponsiveContainer>
              </div>
            </div>

            {/* Plot 2: t-SNE by Best Model Prediction */}
            <div className="bg-black/30 p-6 rounded-2xl border border-white/5 relative">
              <div className="flex items-center justify-between mb-6">
                <div className="text-[8px] font-black text-purple-500 tracking-widest">
                  t-SNE: {tsneData?.best_model || 'Best Model'} Consensus
                </div>
                <div className="h-px flex-1 bg-gradient-to-r from-purple-500/20 to-transparent ml-4" />
              </div>
              <div className="h-[350px]">
                <ResponsiveContainer width="100%" height="100%">
                  <ScatterChart margin={{ top: 10, right: 10, bottom: 10, left: 10 }}>
                    <CartesianGrid
                      strokeDasharray="3 3"
                      stroke="#1f2937"
                      vertical={false}
                      strokeOpacity={0.1}
                    />
                    <XAxis type="number" dataKey="x" hide />
                    <YAxis type="number" dataKey="y" hide />
                    <ZAxis type="number" range={[15, 16]} />
                    <Tooltip
                      cursor={{ strokeDasharray: '3 3' }}
                      content={({ active, payload }) => {
                        if (active && payload && payload.length) {
                          const data = payload[0].payload
                          return (
                            <div className="bg-black/90 border border-white/10 p-2 rounded shadow-xl">
                              <p className="text-[7px] font-black text-purple-500 tracking-tighter">
                                Neural Verdict
                              </p>
                            </div>
                          )
                        }
                        return null
                      }}
                    />
                    <Scatter data={tsneData?.points || []} isAnimationActive={false}>
                      {tsneData?.points?.map((entry, index) => (
                        <Cell
                          key={`cell-${index}`}
                          fill={entry.predicted === 0 ? '#22c55e' : '#f97316'}
                          fillOpacity={0.7}
                        />
                      ))}
                    </Scatter>
                  </ScatterChart>
                </ResponsiveContainer>
              </div>
            </div>

            {/* Plot 3: PCA (Linear Proj) */}
            <div className="bg-black/30 p-6 rounded-2xl border border-white/5 relative">
              <div className="flex items-center justify-between mb-6">
                <div className="text-[8px] font-black text-yellow-500 tracking-widest">
                  PCA: Linear Variance (
                  {(tsneData?.pca_explained_variance?.reduce((a, b) => a + b, 0) * 100)?.toFixed(1)}
                  %)
                </div>
                <div className="h-px flex-1 bg-gradient-to-r from-yellow-500/20 to-transparent ml-4" />
              </div>
              <div className="h-[350px]">
                <ResponsiveContainer width="100%" height="100%">
                  <ScatterChart margin={{ top: 10, right: 10, bottom: 10, left: 10 }}>
                    <CartesianGrid
                      strokeDasharray="3 3"
                      stroke="#1f2937"
                      vertical={false}
                      strokeOpacity={0.1}
                    />
                    <XAxis type="number" dataKey="pca_x" hide />
                    <YAxis type="number" dataKey="pca_y" hide />
                    <ZAxis type="number" range={[15, 16]} />
                    <Tooltip
                      cursor={{ strokeDasharray: '3 3' }}
                      content={({ active, payload }) => {
                        if (active && payload && payload.length) {
                          const data = payload[0].payload
                          return (
                            <div className="bg-black/90 border border-white/10 p-2 rounded shadow-xl">
                              <p className="text-[7px] font-black text-yellow-500 tracking-tighter">
                                Linear Projection
                              </p>
                            </div>
                          )
                        }
                        return null
                      }}
                    />
                    <Scatter data={tsneData?.points || []} isAnimationActive={false}>
                      {tsneData?.points?.map((entry, index) => (
                        <Cell
                          key={`cell-${index}`}
                          fill={entry.true_label === 0 ? '#3b82f6' : '#ef4444'}
                          fillOpacity={0.7}
                        />
                      ))}
                    </Scatter>
                  </ScatterChart>
                </ResponsiveContainer>
              </div>
            </div>

            {/* Plot 4: Risk Heatmap (t-SNE colored by probability) */}
            <div className="bg-black/30 p-6 rounded-2xl border border-white/5 relative">
              <div className="flex items-center justify-between mb-6">
                <div className="text-[8px] font-black text-teal-500 tracking-widest">
                  t-SNE: Neural Risk Topography
                </div>
                <div className="h-px flex-1 bg-gradient-to-r from-teal-500/20 to-transparent ml-4" />
              </div>
              <div className="h-[350px]">
                <ResponsiveContainer width="100%" height="100%">
                  <ScatterChart margin={{ top: 10, right: 10, bottom: 10, left: 10 }}>
                    <CartesianGrid
                      strokeDasharray="3 3"
                      stroke="#1f2937"
                      vertical={false}
                      strokeOpacity={0.1}
                    />
                    <XAxis type="number" dataKey="x" hide />
                    <YAxis type="number" dataKey="y" hide />
                    <ZAxis type="number" range={[15, 16]} />
                    <Tooltip
                      cursor={{ strokeDasharray: '3 3' }}
                      content={({ active, payload }) => {
                        if (active && payload && payload.length) {
                          const data = payload[0].payload
                          return (
                            <div className="bg-black/90 border border-white/10 p-2 rounded shadow-xl">
                              <p className="text-[7px] font-black text-teal-500 tracking-tighter">
                                Risk Gradient
                              </p>
                              <p className="text-[7px] text-white/50 font-bold">
                                Score: {(data.probability * 100).toFixed(1)}%
                              </p>
                            </div>
                          )
                        }
                        return null
                      }}
                    />
                    <Scatter data={tsneData?.points || []} isAnimationActive={false}>
                      {tsneData?.points?.map((entry, index) => (
                        <Cell
                          key={`cell-${index}`}
                          fill={getProbColor(entry.probability)}
                          fillOpacity={0.7}
                        />
                      ))}
                    </Scatter>
                  </ScatterChart>
                </ResponsiveContainer>
              </div>
            </div>
          </div>
        )}
      </AnalyticView>
        )
      })()}
      </div>
  )
}

export default Tsne
