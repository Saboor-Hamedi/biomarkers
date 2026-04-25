import React, { useState, useRef, useEffect } from 'react';
import { MessageSquare, X, Send, Bot, User } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import { cn } from '../lib/utils';
import { getSetting } from '../lib/settings';
import { queryDeepSeek } from '../lib/deepseek';

const ChatBot = ({ appState }) => {
  const [isOpen, setIsOpen] = useState(false);
  const [messages, setMessages] = useState([
    { role: 'bot', text: 'Forensic AI Assistant initialized. How can I help you analyze the current neural trajectory?' }
  ]);
  const [input, setInput] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isTyping, isOpen]);

  const handleSend = async (e) => {
    e.preventDefault();
    if (!input.trim() || isTyping) return;
    
    const userMsg = { role: 'user', content: input };
    const displayUserMsg = { role: 'user', text: input };
    setMessages(prev => [...prev, displayUserMsg]);
    setInput('');
    setIsTyping(true);
    
    try {
      const apiKey = getSetting('DEEPSEEK_API_KEY', import.meta.env.VITE_DEEPSEEK_API_KEY || '');
      
      if (!apiKey || apiKey === 'your_deepseek_api_key_here') {
        setMessages(prev => [...prev, { role: 'bot', text: 'Error: DeepSeek API key not configured. Please click "Settings" in the main sidebar to configure your access token.' }]);
        setIsTyping(false);
        return;
      }

      // Send the entire chat history AND the live appState to the DeepSeek reasoning engine
      const botReply = await queryDeepSeek([...messages, userMsg], appState, apiKey);

      setMessages(prev => [...prev, { role: 'bot', text: botReply }]);
    } catch (error) {
      console.error(error);
      setMessages(prev => [...prev, { role: 'bot', text: `Connection to DeepSeek Core failed: ${error.message}. Please verify your API key and account balance.` }]);
    } finally {
      setIsTyping(false);
    }
  };

  return (
    <>
      {/* Floating Action Button */}
      <button
        onClick={() => setIsOpen(true)}
        className={cn(
          "fixed bottom-6 right-6 p-3 rounded-full bg-blue-600 hover:bg-blue-500 text-white shadow-[0_0_15px_rgba(37,99,235,0.4)] transition-all duration-300 z-50",
          isOpen ? "scale-0 opacity-0 pointer-events-none" : "scale-100 opacity-100 hover:scale-110"
        )}
      >
        <MessageSquare size={18} />
      </button>

      {/* Chat Window */}
      <div 
        className={cn(
          "fixed bottom-6 right-6 w-80 sm:w-96 bg-[#0d1117] border border-gray-800 rounded-lg shadow-2xl flex flex-col transition-all duration-300 transform origin-bottom-right z-50",
          isOpen ? "scale-100 opacity-100" : "scale-0 opacity-0 pointer-events-none"
        )}
        style={{ height: '500px' }}
      >
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b border-gray-800 bg-black/40 rounded-t-lg">
          <div className="flex items-center gap-2">
            <Bot size={18} className="text-blue-500" />
            <h3 className="text-sm font-black uppercase tracking-widest text-white">Forensic Co-Pilot</h3>
          </div>
          <button 
            onClick={() => setIsOpen(false)}
            className="text-gray-500 hover:text-white transition-colors p-1"
          >
            <X size={16} />
          </button>
        </div>

        {/* Message Area */}
        <div className="flex-1 overflow-y-auto p-4 space-y-4 custom-scrollbar">
          {messages.map((msg, idx) => (
            <div key={idx} className={cn("flex gap-3", msg.role === 'user' ? "flex-row-reverse" : "flex-row")}>
              <div className={cn(
                "w-6 h-6 shrink-0 rounded flex items-center justify-center mt-1",
                msg.role === 'user' ? "bg-gray-800 text-gray-300" : "bg-blue-900/50 text-blue-400 border border-blue-500/20"
              )}>
                {msg.role === 'user' ? <User size={12} /> : <Bot size={12} />}
              </div>
              <div className={cn(
                "px-3 py-2 rounded-lg text-[11px] leading-relaxed max-w-[85%]",
                msg.role === 'user' ? "bg-blue-600 text-white rounded-tr-none" : "bg-gray-800/50 text-gray-300 border border-gray-800 rounded-tl-none whitespace-pre-wrap"
              )}>
                <ReactMarkdown
                  components={{
                    p: ({node, ...props}) => <p className="mb-2 last:mb-0" {...props} />,
                    strong: ({node, ...props}) => <strong className={cn("font-bold", msg.role === 'user' ? "text-white" : "text-blue-400")} {...props} />,
                    em: ({node, ...props}) => <em className={cn("italic", msg.role === 'user' ? "text-blue-100" : "text-gray-400")} {...props} />,
                    ul: ({node, ...props}) => <ul className="list-disc pl-4 mb-2 space-y-1 marker:text-gray-600" {...props} />,
                    ol: ({node, ...props}) => <ol className="list-decimal pl-4 mb-2 space-y-1 marker:text-gray-600" {...props} />,
                    li: ({node, ...props}) => <li {...props} />,
                    code: ({node, inline, ...props}) => inline 
                      ? <code className="bg-black/50 px-1 py-0.5 rounded font-mono text-[10px]" {...props} />
                      : <code className="block bg-black/50 p-2 rounded font-mono text-[10px] mb-2 overflow-x-auto whitespace-pre custom-scrollbar" {...props} />
                  }}
                >
                  {msg.text}
                </ReactMarkdown>
              </div>
            </div>
          ))}
          {isTyping && (
            <div className="flex gap-3 flex-row">
              <div className="w-6 h-6 shrink-0 rounded flex items-center justify-center mt-1 bg-blue-900/50 text-blue-400 border border-blue-500/20">
                <Bot size={12} />
              </div>
              <div className="px-3 py-2 rounded-lg text-[11px] leading-relaxed bg-gray-800/50 text-gray-300 border border-gray-800 rounded-tl-none flex items-center gap-1">
                <span className="w-1.5 h-1.5 bg-gray-500 rounded-full animate-bounce"></span>
                <span className="w-1.5 h-1.5 bg-gray-500 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></span>
                <span className="w-1.5 h-1.5 bg-gray-500 rounded-full animate-bounce" style={{ animationDelay: '0.4s' }}></span>
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>

        {/* Input Area */}
        <div className="p-3 border-t border-gray-800 bg-black/20 rounded-b-lg">
          <form onSubmit={handleSend} className="relative">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Ask the neural model..."
              className="w-full bg-[#0d1117] border border-gray-700 rounded text-xs text-white px-3 py-2.5 pr-10 focus:outline-none focus:border-blue-500 transition-colors placeholder:text-gray-600"
            />
            <button 
              type="submit"
              className="absolute right-1.5 top-1.5 p-1 text-blue-500 hover:text-blue-400 transition-colors"
            >
              <Send size={14} />
            </button>
          </form>
        </div>
      </div>
    </>
  );
};

export default ChatBot;
