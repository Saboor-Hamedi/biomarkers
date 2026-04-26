export const queryDeepSeek = async (messages, appState, apiKey) => {
  // Construct dynamic system prompt based on the live application state
  let systemPrompt = `You are a world-class Forensic AI Diagnostic Assistant embedded inside the "Cancer Biomarker AI Suite".
You exist to actively instruct and guide the clinician through complex neural trajectories.
You have FULL visibility into every analytical engine running in the background. Read the provided diagnostic telemetry carefully.
Be concise, highly professional, clinical, and aggressively data-driven. Instruct the user on what the data means.

### CURRENT SYSTEM STATE CONTEXT ###
- Active Dashboard View: ${appState?.activeTab || 'Unknown'}
- Engine Status: ${appState?.engineStatus || 'Unknown'}
`;

  // Inject current patient inputs if available
  if (appState?.inputs) {
    systemPrompt += `
### CURRENT PATIENT BIOMARKER INPUTS ###
- AFP (Alpha-fetoprotein): ${appState.inputs.AFP_pg_per_ml || 0} pg/ml
- CA125: ${appState.inputs.CA125_U_per_ml || 0} U/ml
- PSA: ${appState.inputs.PSA_pg_per_ml || 0} pg/ml
`;
  }

  // Inject prediction results if available
  if (appState?.prediction) {
    systemPrompt += `
### CURRENT NEURAL NETWORK VERDICT ###
- Predicted Risk Probability: ${(appState.prediction.risk_probability * 100).toFixed(2)}%
- Verdict: ${appState.prediction.verdict}
- Highest Impact Feature: ${appState.prediction.dominant_feature}
- Model Consensus Status: ${appState.prediction.consensus_achieved ? 'Achieved' : 'Diverging'}
`;
  } else {
    systemPrompt += `\n*No patient prediction has been run yet. Advise the user to input data in the Forensic Input panel.*\n`;
  }

  // Deep Analytical Results Injection
  if (appState?.metrics) {
    // metrics is an object with {roc, pr, calibration, cm}
    // We'll extract the AUC scores for each model to provide context to the AI
    const committeeSummary = Object.entries(appState.metrics.roc || {}).map(([name, data]) => ({
      model: name,
      auc: data.auc
    }));

    systemPrompt += `
### COMMITTEE PERFORMANCE METRICS ###
The following is the live performance data of our underlying committee of models (AUC scores):
${JSON.stringify(committeeSummary)}
`;
  }

  if (appState?.counterfactualData) {
    systemPrompt += `
### WHAT-IF ENGINE COUNTERFACTUAL PROJECTION ###
If the user asks "how do we lower the risk?" or "what if?", use this exact AI projection:
"${appState.counterfactualData.statement}"
`;
  }

  if (appState?.shapData && appState.shapData.length > 0) {
    systemPrompt += `
### SHAP WATERFALL PATIENT LOGIC ###
These are the exact numerical impacts pulling the patient's risk up or down from the baseline:
${JSON.stringify(appState.shapData)}
`;
  }

  systemPrompt += `
### INSTRUCTIONS FOR AI ###
Use the profound telemetry above to provide highly specific, tailored advice. 
1. If the user asks about the patient, rely entirely on the exact numbers provided in the context above.
2. Instruct the user on what to look for next based on their active view or the counterfactual data.
3. Use markdown formatting (bolding, lists) to make your clinical analysis easy to scan.
`;

  // Map internal UI roles to OpenAI schema ('bot' -> 'assistant')
  const apiMessages = [
    { role: 'system', content: systemPrompt },
    ...messages.filter(m => m.role !== 'system').map(m => ({ 
      role: m.role === 'bot' ? 'assistant' : m.role, 
      content: m.text || m.content 
    }))
  ];

  const response = await fetch('https://api.deepseek.com/chat/completions', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${apiKey}`
    },
    body: JSON.stringify({
      model: 'deepseek-chat',
      messages: apiMessages,
      temperature: 0.2, // Low temperature for highly analytical/clinical responses
      max_tokens: 800
    })
  });

  if (!response.ok) {
    const errData = await response.json().catch(() => ({}));
    throw new Error(errData.error?.message || `DeepSeek API returned status ${response.status}`);
  }

  const data = await response.json();
  return data.choices[0].message.content;
};
