import React, { useState, useEffect, useRef } from 'react'
import {
  Terminal,
  Play,
  ShieldAlert,
  CheckCircle2,
  XCircle,
  Plus,
  Trash2,
  FolderTree,
  FileCode,
  Sparkles,
  ChevronRight,
  Cpu,
  Layers,
  Code2,
  RefreshCw,
  ExternalLink,
  GitBranch,
  Shield,
  Activity
} from 'lucide-react'

const API_BASE = 'http://localhost:8000'

export default function App() {
  const [sessions, setSessions] = useState([])
  const [activeSessionId, setActiveSessionId] = useState('')
  const [messages, setMessages] = useState([])
  const [prompt, setPrompt] = useState('')
  const [isRunning, setIsRunning] = useState(false)
  const [pendingApproval, setPendingApproval] = useState(null)
  
  // Workspace explorer states
  const [rightPanelTab, setRightPanelTab] = useState('files') // 'files' | 'artifacts' | 'guard'
  const [files, setFiles] = useState([])
  const [selectedFile, setSelectedFile] = useState(null)
  const [fileContent, setFileContent] = useState('')
  const [logs, setLogs] = useState([])

  const chatEndRef = useRef(null)

  useEffect(() => {
    fetchSessions()
    fetchFiles()
    const interval = setInterval(() => {
      fetchSessions()
      fetchFiles()
    }, 2500)
    return () => clearInterval(interval)
  }, [])

  useEffect(() => {
    if (activeSessionId) {
      loadSessionState(activeSessionId)
      const interval = setInterval(() => {
        loadSessionState(activeSessionId)
      }, 2000)
      return () => clearInterval(interval)
    }
  }, [activeSessionId])

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, pendingApproval])

  // --- Session Management ---
  const fetchSessions = async () => {
    try {
      const res = await fetch(`${API_BASE}/sessions`)
      if (res.ok) {
        const data = await res.json()
        setSessions(prev => {
          if (JSON.stringify(prev.map(s => s.id)) !== JSON.stringify(data.map(s => s.id))) {
            return data
          }
          return prev
        })
        if (data.length > 0 && !activeSessionId) {
          setActiveSessionId(data[0].id)
        } else if (data.length === 0 && !activeSessionId) {
          handleCreateSession('Default Session')
        }
      }
    } catch (err) {
      // server offline or connecting
    }
  }

  const handleCreateSession = async (title = 'New Agent Task') => {
    try {
      const res = await fetch(`${API_BASE}/sessions`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ title }),
      })
      if (res.ok) {
        const newSession = await res.json()
        setSessions(prev => [newSession, ...prev])
        setActiveSessionId(newSession.id)
        setMessages([
          {
            id: 'init-1',
            sender: 'agent',
            time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
            text: `👋 **Antigravity AI Agent Harness Ready.**\n\n- **Sandbox:** Local Shell Execution (No Docker required)\n- **Guardrail:** Active (\`permissions.json\` HITL Gate)\n- **Model:** Groq / Ollama Hybrid Engine\n\nWhat would you like me to build or inspect today?`,
          }
        ])
      }
    } catch (err) {
      console.error(err)
    }
  }

  const handleDeleteSession = async (id, e) => {
    e.stopPropagation()
    try {
      await fetch(`${API_BASE}/sessions/${id}`, { method: 'DELETE' })
      setSessions(prev => prev.filter(s => s.id !== id))
      if (activeSessionId === id) {
        const remaining = sessions.filter(s => s.id !== id)
        if (remaining.length > 0) {
          setActiveSessionId(remaining[0].id)
        } else {
          handleCreateSession()
        }
      }
    } catch (err) {
      console.error(err)
    }
  }

  const loadSessionState = async (threadId) => {
    try {
      const res = await fetch(`${API_BASE}/runs/${threadId}`)
      if (res.ok) {
        const state = await res.json()
        if (state && state.messages && state.messages.length > 0) {
          // Map LangGraph state messages
          const mapped = state.messages.map((m, idx) => ({
            id: `msg-${idx}`,
            sender: (m.type === 'human' || m.role === 'user') ? 'user' : 'agent',
            time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
            text: typeof m.content === 'string' ? m.content : JSON.stringify(m.content),
          }))
          setMessages(prev => {
            if (prev.length !== mapped.length) return mapped
            const lastPrev = prev[prev.length - 1]?.text
            const lastMapped = mapped[mapped.length - 1]?.text
            if (lastPrev !== lastMapped) return mapped
            return prev
          })
        }
      }
    } catch (err) {
      // No state yet for thread
    }
  }

  // --- Run Execution & HITL Guard ---
  const handleSendMessage = async (e) => {
    e?.preventDefault()
    if (!prompt.trim() || isRunning) return

    const userText = prompt
    setPrompt('')
    const userMsg = {
      id: `user-${Date.now()}`,
      sender: 'user',
      time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      text: userText,
    }
    setMessages(prev => [...prev, userMsg])
    setIsRunning(true)

    try {
      const res = await fetch(`${API_BASE}/runs`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          user_request: userText,
          thread_id: activeSessionId,
        }),
      })

      const data = await res.json()
      setIsRunning(false)

      if (data.status === 'interrupted' || data.ask_approval) {
        setPendingApproval({
          threadId: activeSessionId,
          description: data.description || 'Action triggered an "ask" rule in permissions.json',
          action: data.action || 'Shell / File modification',
        })
      } else {
        const agentMsg = {
          id: `agent-${Date.now()}`,
          sender: 'agent',
          time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
          text: data.output || data.response || (typeof data === 'object' ? JSON.stringify(data, null, 2) : String(data)),
        }
        setMessages(prev => [...prev, agentMsg])
      }
      fetchFiles()
    } catch (err) {
      setIsRunning(false)
      setMessages(prev => [
        ...prev,
        {
          id: `err-${Date.now()}`,
          sender: 'agent',
          time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
          text: `⚠️ **Request Execution Error:** ${err.message}`,
        },
      ])
    }
  }

  const handleApproval = async (approved) => {
    if (!pendingApproval) return
    const threadId = pendingApproval.threadId
    setPendingApproval(null)
    setIsRunning(true)

    try {
      const res = await fetch(`${API_BASE}/runs/${threadId}/resume`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ approved }),
      })
      const data = await res.json()
      setIsRunning(false)

      setMessages(prev => [
        ...prev,
        {
          id: `approval-${Date.now()}`,
          sender: 'agent',
          time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
          text: approved ? `✅ **Action Approved.** Harness proceeded with execution.\n\n${JSON.stringify(data, null, 2)}` : `❌ **Action Rejected by User.** Agent pipeline halted safely.`,
        }
      ])
      fetchFiles()
    } catch (err) {
      setIsRunning(false)
      console.error(err)
    }
  }

  // --- Workspace & Code Inspector ---
  const fetchFiles = async () => {
    try {
      const res = await fetch(`${API_BASE}/workspace/files`)
      if (res.ok) {
        const data = await res.json()
        setFiles(data.files || [])
      }
    } catch (err) {
      console.log('Workspace files fetch error:', err)
    }
  }

  const handleSelectFile = async (file) => {
    if (file.is_dir) return
    setSelectedFile(file)
    try {
      const res = await fetch(`${API_BASE}/workspace/file?path=${encodeURIComponent(file.path)}`)
      if (res.ok) {
        const data = await res.json()
        setFileContent(data.content)
      }
    } catch (err) {
      setFileContent(`Error loading file: ${err.message}`)
    }
  }

  return (
    <div className="app-container">
      {/* 1. Left Sidebar: Antigravity Session Tree & Workspaces */}
      <aside className="sidebar">
        <div className="sidebar-header">
          <div className="logo-badge">
            <div className="logo-icon-wrap">
              <Sparkles size={16} />
            </div>
            <span>ANTIGRAVITY</span>
          </div>
          <span className="badge-tag">v2.0</span>
        </div>

        <button className="btn-new-chat" onClick={() => handleCreateSession()}>
          <Plus size={15} />
          <span>New Session</span>
        </button>

        <div className="session-list">
          <div style={{ fontSize: '11px', color: 'var(--text-dim)', padding: '6px 12px', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.05em' }}>
            Active Threads
          </div>
          {sessions.map(s => (
            <div
              key={s.id}
              className={`session-item ${activeSessionId === s.id ? 'active' : ''}`}
              onClick={() => setActiveSessionId(s.id)}
            >
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <Code2 size={14} style={{ color: activeSessionId === s.id ? 'var(--accent-cyan)' : 'var(--text-dim)' }} />
                <span className="session-title">{s.title || `Thread ${s.id.slice(0, 8)}`}</span>
              </div>
              <button className="session-delete-btn" onClick={(e) => handleDeleteSession(s.id, e)} title="Delete session">
                <Trash2 size={13} />
              </button>
            </div>
          ))}
        </div>

        <div className="sidebar-footer">
          <div className="status-indicator">
            <span className="status-dot"></span>
            <span>Harness Guard Active</span>
          </div>
          <Shield size={14} style={{ color: 'var(--accent-cyan)' }} />
        </div>
      </aside>

      {/* 2. Center Stage: Agent Command & Execution Timeline */}
      <main className="main-stage">
        <header className="stage-header">
          <div className="stage-title-wrap">
            <div className="stage-title">
              {sessions.find(s => s.id === activeSessionId)?.title || 'Coding Harness'}
            </div>
            <span className="badge-tag">LOCAL-SANDBOX</span>
            <span className="badge-tag" style={{ color: 'var(--accent-green)', borderColor: 'rgba(16,185,129,0.3)', background: 'rgba(16,185,129,0.1)' }}>
              HITL ENABLED
            </span>
          </div>

          <div className="stage-actions">
            <button
              className={`action-btn ${rightPanelTab === 'files' ? 'active' : ''}`}
              onClick={() => setRightPanelTab(rightPanelTab === 'files' ? null : 'files')}
            >
              <FolderTree size={14} />
              <span>Workspace</span>
            </button>
          </div>
        </header>

        {/* Message Stream */}
        <div className="chat-scroll">
          {messages.map(msg => (
            <div key={msg.id} className="message-item">
              <div className={`message-avatar ${msg.sender === 'user' ? 'avatar-user' : 'avatar-agent'}`}>
                {msg.sender === 'user' ? 'U' : <Sparkles size={16} />}
              </div>
              <div className="message-body">
                <div className="message-meta">
                  <span className="message-sender">{msg.sender === 'user' ? 'Developer' : 'Antigravity Agent'}</span>
                  <span className="message-time">{msg.time}</span>
                </div>
                <div className="message-bubble">
                  <div style={{ whiteSpace: 'pre-wrap', wordBreak: 'break-word' }}>
                    {msg.text}
                  </div>
                </div>
              </div>
            </div>
          ))}

          {/* Pending HITL Approval Gate Card */}
          {pendingApproval && (
            <div className="guard-card">
              <div className="guard-header">
                <ShieldAlert size={18} />
                <span>HUMAN-IN-THE-LOOP APPROVAL REQUIRED</span>
              </div>
              <div className="guard-desc">
                {pendingApproval.description}
                <div style={{ fontFamily: 'var(--font-mono)', fontSize: '11.5px', marginTop: '6px', color: '#f1f5f9', background: 'rgba(0,0,0,0.4)', padding: '6px 10px', borderRadius: '4px' }}>
                  Action: {pendingApproval.action}
                </div>
              </div>
              <div className="guard-actions">
                <button className="btn-approve" onClick={() => handleApproval(true)}>
                  <CheckCircle2 size={14} />
                  <span>Approve & Proceed</span>
                </button>
                <button className="btn-reject" onClick={() => handleApproval(false)}>
                  <XCircle size={14} />
                  <span>Reject</span>
                </button>
              </div>
            </div>
          )}

          {isRunning && (
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px', color: 'var(--accent-cyan)', fontSize: '12.5px', padding: '10px 0' }}>
              <RefreshCw size={14} className="spin" style={{ animation: 'spin 1s linear infinite' }} />
              <span>Autonomous agent planning and executing commands...</span>
            </div>
          )}

          <div ref={chatEndRef} />
        </div>

        {/* Input Dock */}
        <div className="input-dock">
          <form className="input-box" onSubmit={handleSendMessage}>
            <textarea
              className="prompt-textarea"
              placeholder="Ask Antigravity to build, debug, refactor, or test (e.g. 'Build an authentication API with tests')..."
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault()
                  handleSendMessage()
                }
              }}
            />
            <div className="input-controls">
              <div className="model-pill">
                <Cpu size={12} style={{ color: 'var(--accent-cyan)' }} />
                <span>qwen/qwen3.8-27b (Groq)</span>
              </div>
              <button
                type="submit"
                className="send-btn"
                disabled={!prompt.trim() || isRunning}
                title="Send Command"
              >
                <Play size={14} fill="currentColor" />
              </button>
            </div>
          </form>
        </div>
      </main>

      {/* 3. Right Panel: Workspace Explorer & Code Preview */}
      {rightPanelTab && (
        <aside className="workspace-panel">
          <div className="panel-header">
            <div className="panel-tabs">
              <button
                className={`panel-tab ${selectedFile ? '' : 'active'}`}
                onClick={() => setSelectedFile(null)}
              >
                Explorer
              </button>
              {selectedFile && (
                <button className="panel-tab active">
                  {selectedFile.name}
                </button>
              )}
            </div>
            <button className="action-btn" onClick={fetchFiles} title="Refresh Files">
              <RefreshCw size={12} />
            </button>
          </div>

          <div className="panel-content">
            {!selectedFile ? (
              <div>
                <div style={{ fontSize: '11px', color: 'var(--text-dim)', padding: '4px 6px', fontWeight: 600, textTransform: 'uppercase' }}>
                  Workspace Files
                </div>
                {files.map(f => (
                  <div
                    key={f.path}
                    className="file-tree-item"
                    onClick={() => handleSelectFile(f)}
                  >
                    {f.is_dir ? <FolderTree size={14} style={{ color: 'var(--accent-amber)' }} /> : <FileCode size={14} style={{ color: 'var(--accent-cyan)' }} />}
                    <span>{f.name}</span>
                  </div>
                ))}
              </div>
            ) : (
              <div className="code-viewer-container">
                <div className="code-viewer-header">
                  <span>{selectedFile.path}</span>
                  <button
                    style={{ background: 'none', border: 'none', color: 'var(--text-dim)', cursor: 'pointer' }}
                    onClick={() => setSelectedFile(null)}
                  >
                    Close
                  </button>
                </div>
                <div className="code-viewer-body">
                  {fileContent}
                </div>
              </div>
            )}
          </div>
        </aside>
      )}

      <style>{`
        @keyframes spin {
          from { transform: rotate(0deg); }
          to { transform: rotate(360deg); }
        }
      `}</style>
    </div>
  )
}
