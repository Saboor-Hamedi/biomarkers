import React from 'react'

const CommitteeReview = () => {
  // Hardcoded historical performance from result.md for the Best Model (XGBoost)
  const bestModelName = "XGBoost"
  const metrics = [
    { label: 'Accuracy', value: '94.67%' },
    { label: 'Precision', value: '75.00%' },
    { label: 'Recall', value: '83.33%' },
    { label: 'F1-Score', value: '78.95%' },
    { label: 'ROC-AUC', value: '98.71%' }
  ]

  return (
    <div className="bg-[#0d1117] rounded-lg p-6 border border-gray-800">
      <h2 className="text-[14px] font-bold text-white leading-tight mb-1">
        Highest Scoring Model: {bestModelName}
      </h2>
      <p className="text-[11px] text-gray-500 mb-6">
        Top performance across validation metrics
      </p>

      <div className="space-y-0">
        {metrics.map((metric, i) => (
          <div key={i}>
            <div className="flex items-center justify-between py-3">
              <span className="text-[12px] font-bold text-gray-300">
                {metric.label}
              </span>
              <span className="text-[12px] font-bold text-blue-400">
                {metric.value}
              </span>
            </div>
            {i !== metrics.length - 1 && (
              <div className="h-[1px] w-full bg-gray-800" />
            )}
          </div>
        ))}
      </div>

      <p className="text-[9px] text-gray-600 mt-6 leading-relaxed">
        This figure highlights only the best performing model based on F1 score, providing a concise single-model summary.
      </p>
    </div>
  )
}

export default CommitteeReview
