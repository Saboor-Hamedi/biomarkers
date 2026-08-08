import React from 'react'

const CommitteeReview = ({ performanceData }) => {
  // Live data from the server /performance endpoint (5-fold CV evaluation).
  // Falls back to a loading / offline message when no data is available.
  const models = Array.isArray(performanceData) ? performanceData : []
  const best = models.find((m) => m.highlight) || models[0] || null

  return (
    <div className="bg-[#0d1117] rounded-lg p-6 border border-gray-800">
      {best ? (
        <>
          <h2 className="text-[14px] font-bold text-white leading-tight mb-1">
            Highest Scoring Model: {best.name}
          </h2>
          <p className="text-[11px] text-gray-500 mb-6">
            Top performance across validation metrics (live evaluation)
          </p>

          <div className="space-y-0">
            {[
              { label: 'Accuracy', value: best.acc },
              { label: 'Precision', value: best.prec },
              { label: 'Recall', value: best.rec },
              { label: 'F1-Score', value: best.f1 },
              { label: 'ROC-AUC', value: best.roc }
            ].map((metric, i) => (
              <div key={i}>
                <div className="flex items-center justify-between py-3">
                  <span className="text-[12px] font-bold text-gray-300">
                    {metric.label}
                  </span>
                  <span className="text-[12px] font-bold text-blue-400">
                    {metric.value}%
                  </span>
                </div>
                {i !== 4 && <div className="h-[1px] w-full bg-gray-800" />}
              </div>
            ))}
          </div>

          <p className="text-[9px] text-gray-600 mt-6 leading-relaxed">
            Best model selected dynamically from the evaluated committee based on F1 score.
          </p>
        </>
      ) : (
        <div className="py-12 flex flex-col items-center justify-center text-center opacity-30">
          <h2 className="text-[14px] font-bold text-white mb-2">Committee Review</h2>
          <p className="text-[10px] font-bold text-gray-500 tracking-widest">
            {models.length === 0 ? 'Audit Required — no evaluation data yet' : 'Loading live metrics...'}
          </p>
        </div>
      )}
    </div>
  )
}

export default CommitteeReview
