export const queryDeepSeek = async (messages, appState, apiKey) => {
  // Construct dynamic system prompt based on the live application state
  let systemPrompt = `You are a world-class Forensic AI Diagnostic Assistant embedded inside the "Cancer Biomarker AI Suite".
You exist to actively instruct and guide the clinician through complex neural trajectories.
### CRITICAL DIRECTIVE ###
You have FULL visibility into every analytical engine running in the background. Read the provided diagnostic telemetry carefully.
The telemetry is LIVE. If the current telemetry contradicts earlier messages in this chat history, it means the user has updated the system or loaded a new patient. ALWAYS trust the CURRENT SYSTEM STATE over the chat history.
Be concise, highly professional, clinical, and aggressively data-driven. Instruct the user on what the data means.

### CURRENT SYSTEM STATE CONTEXT ###
- Active Dashboard View: ${appState?.activeTab || 'Unknown'}
- Engine Status: ${appState?.engineStatus || 'Unknown'}

### CLINICAL KNOWLEDGE BASE ###
You are analyzing Prostate Cancer Risk using DPV voltammetry features combined with traditional biomarkers.
- **Biomarkers**: PSA (Prostate-Specific Antigen) > 4 ng/mL is the standard clinical threshold for elevated risk. AFP and CA125 are secondary tumor markers.
- **The Neural Ensemble**: The system uses a consensus of 5 models: XGBoost (historically highest accuracy at 94.67%), Support Vector Machine (SVM), Random Forest, Logistic Regression, and a Graph Neural Network (GNN).
- **SHAP Values**: SHAP (SHapley Additive exPlanations) quantifies feature impact. A positive SHAP value pushes the patient's risk higher (towards malignancy); a negative SHAP value pulls the risk lower (towards benign).
- **Counterfactuals**: The "What-If" engine calculates the absolute minimum physiological change required to flip a malignant prediction to benign.

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
- Predicted Risk Score: ${appState.prediction.risk_score} (Probability: ${(appState.prediction.risk_score * 100).toFixed(2)}%)
- Verdict: ${appState.prediction.prediction}
- Model Consensus: ${appState.prediction.consensus}

### INDIVIDUAL MODEL PROBABILITIES ###
${appState.prediction.models ? Object.entries(appState.prediction.models).map(([model, prob]) => `- ${model}: ${prob}`).join('\n') : '*No individual model breakdown available*'}
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
### STRICT FORMATTING INSTRUCTIONS FOR AI ###
1. **LEAD WITH THE VERDICT**: Always begin your analysis by explicitly stating the main Risk Score, overall Verdict, and Consensus percentage. NEVER guess or hallucinate these numbers. If they are provided in the telemetry, use them exactly. If they are not provided, explicitly state that no prediction is loaded.
2. **BE EXTREMELY CONCISE**: Limit your response to 2-3 short, punchy paragraphs. Do not write essays.
3. **NO RAW DATA/JSON**: Never spit out raw JSON or massive data dumps. Translate the telemetry and numerical data into plain, clinical English bullet points.
4. **CLINICAL TONE**: Be highly professional and aggressively data-driven. Use strong verbs.
5. **MARKDOWN ONLY**: Use bolding for key metrics (e.g., **99.5%**) and bullet points to make your analysis instantly scannable by a busy clinician.
6. **DIRECT ANSWERS**: Do not use filler phrases like "Based on the provided telemetry...". Just answer the question immediately.
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
