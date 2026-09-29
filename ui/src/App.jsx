import React, { useState, useEffect, useRef } from 'react'
import {
  Terminal,
  Play,
  ShieldAlert,
  CheckCircle2,
  XCircle,
  Square,
  Plus,
  Trash2,
  FolderTree,
  FileCode,
  Sparkles,
  ChevronRight,
  ChevronDown,
  Folder,
  FolderOpen,
  Cpu,
  Layers,
  Code2,
  RefreshCw,
  ExternalLink,
  GitBranch,
  Shield,
  Activity,
  Paperclip,
  Image as ImageIcon,
  X,
  File
} from 'lucide-react'

const API_BASE = 'http://127.0.0.1:8000'

const getWelcomeMessage = () => ({
  id: 'init-1',
  sender: 'agent',
  time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
  text: `👋 **AI Coding Agent Ready.**\n\n- **Sandbox:** Local Shell Execution (No Docker required)\n- **Guardrail:** Active (\`permissions.json\` HITL Gate)\n- **Model:** Groq / Ollama Hybrid Engine\n\nWhat would you like me to build or inspect today?`,
})

export default function App() {
  const [sessions, setSessions] = useState([])
  const [activeSessionId, setActiveSessionId] = useState('')
  const [messages, setMessages] = useState([getWelcomeMessage()])
  const [prompt, setPrompt] = useState('')
  const [isRunning, setIsRunning] = useState(false)
  const [pendingApproval, setPendingApproval] = useState(null)
  const [attachments, setAttachments] = useState([])
  const [selectedModel, setSelectedModel] = useState('groq:qwen/qwen3.8-27b')
  const fileInputRef = useRef(null)
  
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

  const handleSelectSession = (sessionId) => {
    if (sessionId === activeSessionId) return
    // Clear out previous session messages immediately so they don't bleed into new session
    setMessages([getWelcomeMessage()])
    setPendingApproval(null)
    setAttachments([])
    setActiveSessionId(sessionId)
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
        setMessages([getWelcomeMessage()])
        setPendingApproval(null)
        setAttachments([])
        setActiveSessionId(newSession.id)
      }
    } catch (err) {
      console.error(err)
    }
  }

  const handleFileSelect = (e) => {
    const files = Array.from(e.target.files || [])
    if (!files.length) return

    files.forEach(file => {
      const isImg = file.type.startsWith('image/')
      const reader = new FileReader()

      if (isImg) {
        reader.onload = (event) => {
          setAttachments(prev => [
            ...prev,
            {
              id: `att-${Date.now()}-${Math.random()}`,
              name: file.name,
              type: 'image',
              size: file.size,
              dataUrl: event.target.result,
            }
          ])
        }
        reader.readAsDataURL(file)
      } else {
        // Read as text snippet if possible (code / config / text files)
        reader.onload = (event) => {
          setAttachments(prev => [
            ...prev,
            {
              id: `att-${Date.now()}-${Math.random()}`,
              name: file.name,
              type: 'file',
              size: file.size,
              content: event.target.result,
            }
          ])
        }
        reader.readAsText(file)
      }
    })

    // Reset input so re-selecting same file triggers onChange
    if (e.target) e.target.value = ''
  }

  const handleRemoveAttachment = (id) => {
    setAttachments(prev => prev.filter(a => a.id !== id))
  }

  const handleDeleteSession = async (id, e) => {
    e.stopPropagation()
    try {
      await fetch(`${API_BASE}/sessions/${id}`, { method: 'DELETE' })
      setSessions(prev => prev.filter(s => s.id !== id))
      if (activeSessionId === id) {
        const remaining = sessions.filter(s => s.id !== id)
        if (remaining.length > 0) {
          handleSelectSession(remaining[0].id)
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
          // Map messages whether returned from LangGraph or persistent session JSON
          const mapped = state.messages.map((m, idx) => ({
            id: m.id || `msg-${idx}`,
            sender: m.sender || (m.type === 'human' || m.role === 'user' ? 'user' : 'agent'),
            time: m.time || new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
            text: typeof m.text === 'string'
              ? m.text
              : (typeof m.content === 'string' ? m.content : JSON.stringify(m.content || m)),
            attachments: m.attachments || [],
          }))
          setMessages(prev => {
            if (prev.length !== mapped.length) return mapped
            const lastPrev = prev[prev.length - 1]?.text
            const lastMapped = mapped[mapped.length - 1]?.text
            if (lastPrev !== lastMapped) return mapped
            return prev
          })
          return
        }
      }
      // If thread has no runs or messages recorded yet, ensure clean initial greeting
      setMessages(prev => {
        if (prev.length === 1 && prev[0].id === 'init-1') return prev
        // Only reset if it's currently holding messages from an execution
        if (prev.some(m => m.id !== 'init-1')) {
          return [getWelcomeMessage()]
        }
        return prev
      })
    } catch (err) {
      // No state yet for thread
    }
  }

  const runAbortControllerRef = useRef(null)

  // --- Stop In-Flight Run ---
  const handleStopRun = async () => {
    if (!isRunning) return
    try {
      if (runAbortControllerRef.current) {
        runAbortControllerRef.current.abort()
      }
      if (activeSessionId) {
        await fetch(`${API_BASE}/runs/${activeSessionId}/stop`, { method: 'POST' })
      }
    } catch (e) {
      console.warn('Failed to dispatch stop request', e)
    } finally {
      setIsRunning(false)
      setMessages(prev => [
        ...prev,
        {
          id: `stop-${Date.now()}`,
          sender: 'agent',
          time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
          text: '🛑 **Execution stopped by user.**',
        },
      ])
    }
  }

  // --- Run Execution & HITL Guard ---
  const handleSendMessage = async (e) => {
    e?.preventDefault()
    if (isRunning) {
      handleStopRun()
      return
    }
    if (!prompt.trim() && attachments.length === 0) return

    const rawPrompt = prompt
    const currentAttachments = [...attachments]
    setPrompt('')
    setAttachments([])

    // Construct enriched prompt with attached file contents / names
    let finalPrompt = rawPrompt.trim()
    const fileSnippets = currentAttachments.map(att => {
      if (att.type === 'file' && att.content) {
        return `\n\n--- Attachment: ${att.name} ---\n\`\`\`\n${att.content.slice(0, 10000)}\n\`\`\``
      } else if (att.type === 'image') {
        return `\n\n[Attached image: ${att.name}]`
      }
      return `\n\n[Attached file: ${att.name}]`
    }).join('')

    if (fileSnippets) {
      finalPrompt = `${finalPrompt || 'Please review the attached file(s):'}${fileSnippets}`
    }

    const userMsg = {
      id: `user-${Date.now()}`,
      sender: 'user',
      time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      text: rawPrompt || `Shared ${currentAttachments.length} file(s)`,
      attachments: currentAttachments,
    }
    setMessages(prev => [...prev, userMsg])
    setIsRunning(true)

    const controller = new AbortController()
    runAbortControllerRef.current = controller

    try {
      const res = await fetch(`${API_BASE}/runs`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        signal: controller.signal,
        body: JSON.stringify({
          user_request: finalPrompt,
          thread_id: activeSessionId,
          model: selectedModel,
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
      if (err.name === 'AbortError') {
        return
      }
      const isFetchFail = err.message && err.message.toLowerCase().includes('fetch')
      const errorText = isFetchFail
        ? `⚠️ **Backend Server Offline:** Could not reach the API at \`${API_BASE}\`.\n\n` +
          `Please make sure the API server is running:\n` +
          `1. Open a terminal in the project root\n` +
          `2. Run: \`& ".\\.venv\\Scripts\\python.exe" -m uvicorn interfaces.api_server:app --port 8000 --reload\`\n` +
          `*(Or press \`Ctrl+Shift+P\` → \`AI Coding Agent: Start API Server\`)*`
        : `⚠️ **Request Execution Error:** ${err.message}`

      setMessages(prev => [
        ...prev,
        {
          id: `err-${Date.now()}`,
          sender: 'agent',
          time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
          text: errorText,
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
  const [expandedFolders, setExpandedFolders] = useState({})
  const [folderContents, setFolderContents] = useState({})

  const fetchFiles = async (folderPath = '') => {
    try {
      const url = folderPath 
        ? `${API_BASE}/workspace/files?path=${encodeURIComponent(folderPath)}`
        : `${API_BASE}/workspace/files`
      const res = await fetch(url)
      if (res.ok) {
        const data = await res.json()
        if (!folderPath) {
          setFiles(data.files || [])
        } else {
          setFolderContents(prev => ({ ...prev, [folderPath]: data.files || [] }))
        }
      }
    } catch (err) {
      console.log('Workspace files fetch error:', err)
    }
  }

  const toggleFolder = async (folder) => {
    const isExpanded = !!expandedFolders[folder.path]
    setExpandedFolders(prev => ({ ...prev, [folder.path]: !isExpanded }))
    if (!isExpanded && !folderContents[folder.path]) {
      await fetchFiles(folder.path)
    }
  }

  const handleSelectFile = async (file) => {
    if (file.is_dir) {
      toggleFolder(file)
      return
    }
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

  const renderFileTree = (items, depth = 0) => {
    return items.map(f => {
      const isDir = f.is_dir
      const isExpanded = !!expandedFolders[f.path]
      const children = folderContents[f.path] || []

      return (
        <div key={f.path}>
          <div
            className={`file-tree-item ${selectedFile?.path === f.path ? 'active' : ''}`}
            style={{ paddingLeft: `${8 + depth * 14}px` }}
            onClick={() => handleSelectFile(f)}
          >
            {isDir ? (
              <>
                <span style={{ display: 'inline-flex', alignItems: 'center', opacity: 0.6 }}>
                  {isExpanded ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
                </span>
                {isExpanded ? (
                  <FolderOpen size={14} style={{ color: 'var(--accent-amber)' }} />
                ) : (
                  <Folder size={14} style={{ color: 'var(--accent-amber)' }} />
                )}
              </>
            ) : (
              <>
                <span style={{ width: '12px', display: 'inline-block' }} />
                <FileCode size={14} style={{ color: 'var(--accent-cyan)' }} />
              </>
            )}
            <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
              {f.name}
            </span>
          </div>

          {/* Render children if folder is open */}
          {isDir && isExpanded && (
            <div className="file-tree-subfolder">
              {children.length === 0 ? (
                <div style={{ paddingLeft: `${24 + depth * 14}px`, fontSize: '11px', color: 'var(--text-dim)', fontStyle: 'italic', paddingBottom: '4px' }}>
                  (Empty)
                </div>
              ) : (
                renderFileTree(children, depth + 1)
              )}
            </div>
          )}
        </div>
      )
    })
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
            <span>CODING AGENT</span>
          </div>
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
              onClick={() => handleSelectSession(s.id)}
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
          <Shield size={14} style={{ color: 'var(--text-muted)' }} />
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
                  <span className="message-sender">{msg.sender === 'user' ? 'Developer' : 'AI Agent'}</span>
                  <span className="message-time">{msg.time}</span>
                </div>
                <div className="message-bubble">
                  {msg.attachments && msg.attachments.length > 0 && (
                    <div className="message-attachments-preview">
                      {msg.attachments.map(att => (
                        <div key={att.id} className="message-att-item">
                          {att.type === 'image' ? (
                            <img src={att.dataUrl} alt={att.name} className="message-att-thumb" />
                          ) : (
                            <File size={13} style={{ color: 'var(--accent-cyan)' }} />
                          )}
                          <span className="message-att-name">{att.name}</span>
                        </div>
                      ))}
                    </div>
                  )}
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
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px', color: 'var(--text-main)', fontSize: '12.5px', padding: '10px 0' }}>
              <RefreshCw size={14} className="spin" style={{ animation: 'spin 1s linear infinite' }} />
              <span>Autonomous agent planning and executing commands...</span>
            </div>
          )}

          <div ref={chatEndRef} />
        </div>

        {/* Input Dock */}
        <div className="input-dock">
          <form className="input-box" onSubmit={handleSendMessage}>
            {/* Attachment preview tray */}
            {attachments.length > 0 && (
              <div className="attachments-tray">
                {attachments.map(att => (
                  <div key={att.id} className="attachment-chip">
                    {att.type === 'image' ? (
                      <img src={att.dataUrl} alt={att.name} className="attachment-chip-thumb" />
                    ) : (
                      <File size={13} style={{ color: 'var(--text-muted)' }} />
                    )}
                    <span className="attachment-chip-name">{att.name}</span>
                    <button
                      type="button"
                      className="attachment-remove-btn"
                      onClick={() => handleRemoveAttachment(att.id)}
                      title="Remove"
                    >
                      <X size={12} />
                    </button>
                  </div>
                ))}
              </div>
            )}

            <textarea
              className="prompt-textarea"
              placeholder="Ask the agent to build, debug, refactor, or test (e.g. 'Build an authentication API with tests')..."
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault()
                  handleSendMessage()
                }
              }}
            />

            <input
              type="file"
              ref={fileInputRef}
              style={{ display: 'none' }}
              multiple
              onChange={handleFileSelect}
            />

            <div className="input-controls">
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <button
                  type="button"
                  className="attach-btn"
                  onClick={() => fileInputRef.current?.click()}
                  title="Attach files or images"
                >
                  <Plus size={16} />
                </button>
                <div className="model-pill">
                  <Cpu size={12} style={{ color: 'var(--text-muted)' }} />
                  <select
                    className="model-select"
                    value={selectedModel}
                    onChange={(e) => setSelectedModel(e.target.value)}
                    title="Select AI Model"
                  >
                    <option value="groq:qwen/qwen3.8-27b">qwen/qwen3.8-27b (Groq)</option>
                    <option value="ollama:gpt-oss:120b-cloud">gpt-oss:120b-cloud (Ollama)</option>
                    <option value="ollama:gemma4:cloud">gemma4:cloud (Ollama)</option>
                    <option value="ollama:nemotron-3-super:cloud">nemotron-3-super:cloud (Ollama)</option>
                  </select>
                </div>
              </div>

              <button
                type={isRunning ? 'button' : 'submit'}
                className={`send-btn ${isRunning ? 'stop-btn' : ''}`}
                onClick={isRunning ? handleStopRun : undefined}
                disabled={!isRunning && !prompt.trim() && attachments.length === 0}
                title={isRunning ? 'Stop Execution' : 'Send Command'}
              >
                {isRunning ? (
                  <Square size={13} fill="currentColor" />
                ) : (
                  <Play size={14} fill="currentColor" />
                )}
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
                {renderFileTree(files)}
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
