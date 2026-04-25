export const queryDeepSeek = async (messages, appState, apiKey) => {
  // Construct dynamic system prompt based on the live application state
  let systemPrompt = `You are a world-class Forensic AI Diagnostic Assistant embedded inside the "Cancer Biomarker AI Suite".
You exist to help the clinician analyze the current neural trajectory and patient data.
Be concise, highly professional, clinical, and data-driven in your answers.

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

  systemPrompt += `\nUse the context above to provide highly specific, tailored advice. If the user asks about the patient, rely entirely on the exact numbers provided in the context above.`;

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
