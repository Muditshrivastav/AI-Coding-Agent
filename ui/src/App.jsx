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
  File,
  History,
  Copy,
  Check,
  Bot,
  Eye,
  EyeOff,
  FilePen,
  Search,
  FolderSearch,
  Wrench,
  Zap
} from 'lucide-react'

const API_BASE = 'http://127.0.0.1:8000'

// ── Tool call pattern detector ──────────────────────────────────────────────
// Maps patterns found in agent text to structured tool activity objects.
const TOOL_PATTERNS = [
  {
    id: 'read_file',
    regex: /(?:reading?|opened?|loading?|fetching?|viewing?)\s+(?:file\s+)?[`'"]?([\w./\\:_-]+\.[\w]+)[`'"]?/i,
    label: 'Read File',
    icon: 'eye',
    color: 'cyan',
  },
  {
    id: 'write_file',
    regex: /(?:writing?|creating?|saving?|wrote|created?|generated?)\s+(?:file\s+)?[`'"]?([\w./\\:_-]+\.[\w]+)[`'"]?/i,
    label: 'Write File',
    icon: 'pen',
    color: 'green',
  },
  {
    id: 'edit_file',
    regex: /(?:editing?|modifying?|updating?|patching?|changed?)\s+(?:file\s+)?[`'"]?([\w./\\:_-]+\.[\w]+)[`'"]?/i,
    label: 'Edit File',
    icon: 'pen',
    color: 'amber',
  },
  {
    id: 'bash',
    regex: /(?:running?|executing?|ran|executed?)\s+(?:command\s+|shell\s+|bash\s+)?[`'"]([^`'"\n]{3,80})[`'"]?/i,
    label: 'Shell Command',
    icon: 'terminal',
    color: 'rose',
  },
  {
    id: 'list_dir',
    regex: /(?:listing?|scanning?|exploring?)\s+(?:directory|folder|dir)\s+[`'"]?([\w./\\:_-]*)[`'"]?/i,
    label: 'List Directory',
    icon: 'folder',
    color: 'amber',
  },
  {
    id: 'search',
    regex: /(?:searching?|grep(?:ping)?|finding?|looking for)\s+[`'"]?([^`'"\n]{3,60})[`'"]?/i,
    label: 'Search',
    icon: 'search',
    color: 'blue',
  },
]

/**
 * Splits agent message text into alternating plain-text and tool-activity segments.
 * Returns an array of { type: 'text'|'tool', ... } objects.
 */
function parseMessageParts(text) {
  if (!text) return [{ type: 'text', content: '' }]

  // Detect explicit [TOOL: ...] markers from backend (if present)
  const toolBlockRegex = /\[TOOL(?::([^\]]+))?\]([\s\S]*?)\[\/TOOL\]/g
  const parts = []
  let lastIdx = 0
  let m

  while ((m = toolBlockRegex.exec(text)) !== null) {
    if (m.index > lastIdx) {
      parts.push({ type: 'text', content: text.slice(lastIdx, m.index).trim() })
    }
    const toolName = (m[1] || '').trim().toLowerCase()
    const body = (m[2] || '').trim()
    // Extract a short target from the body (first line, max 80 chars)
    const firstLine = body.split('\n')[0].slice(0, 80)
    const pattern = TOOL_PATTERNS.find(p => p.id === toolName) || {
      id: toolName || 'tool',
      label: toolName || 'Tool Call',
      icon: 'wrench',
      color: 'cyan',
    }
    parts.push({ type: 'tool', tool: pattern, target: firstLine, body })
    lastIdx = m.index + m[0].length
  }

  if (lastIdx < text.length) {
    parts.push({ type: 'text', content: text.slice(lastIdx).trim() })
  }

  // If no explicit markers were found, return as single text block
  if (parts.length === 0 || (parts.length === 1 && parts[0].type === 'text')) {
    return [{ type: 'text', content: text }]
  }

  return parts.filter(p => p.type === 'tool' || (p.type === 'text' && p.content))
}

// ── Tool Activity Card: renders a single SSE tool event ─────────────────────
// Matches Antigravity-style: "Analyzed App.jsx #L100-160" compact chips
const TOOL_META = {
  // deepagents read operations
  read_file:       { label: 'Read',     icon: 'eye',      color: 'cyan'  },
  view_file:       { label: 'Analyzed', icon: 'eye',      color: 'cyan'  },
  read:            { label: 'Analyzed', icon: 'eye',      color: 'cyan'  },
  glob:            { label: 'Listed',   icon: 'folder',   color: 'amber' },
  ls:              { label: 'Listed',   icon: 'folder',   color: 'amber' },
  grep:            { label: 'Searched', icon: 'search',   color: 'blue'  },
  grep_search:     { label: 'Searched', icon: 'search',   color: 'blue'  },
  list_dir:        { label: 'Listed',   icon: 'folder',   color: 'amber' },
  // deepagents write/edit operations
  write:           { label: 'Wrote',    icon: 'pen',      color: 'green' },
  write_file:      { label: 'Wrote',    icon: 'pen',      color: 'green' },
  edit:            { label: 'Edited',   icon: 'pen',      color: 'amber' },
  edit_file:       { label: 'Edited',   icon: 'pen',      color: 'amber' },
  replace_file_content: { label: 'Edited', icon: 'pen',  color: 'amber' },
  write_to_file:   { label: 'Wrote',    icon: 'pen',      color: 'green' },
  create:          { label: 'Created',  icon: 'pen',      color: 'green' },
  delete:          { label: 'Deleted',  icon: 'wrench',   color: 'rose'  },
  // shell
  execute:         { label: 'Ran',      icon: 'terminal', color: 'rose'  },
  execute_command: { label: 'Ran',      icon: 'terminal', color: 'rose'  },
  bash:            { label: 'Ran',      icon: 'terminal', color: 'rose'  },
  run_command:     { label: 'Ran',      icon: 'terminal', color: 'rose'  },
  run_verification:{ label: 'Verified', icon: 'terminal', color: 'green' },
  // search/web
  tavily_web_search:{ label: 'Searched', icon: 'search',  color: 'blue' },
  tavily_search_results_json: { label: 'Searched', icon: 'search', color: 'blue' },
  // misc
  probabilistic_decision: { label: 'Decided', icon: 'wrench', color: 'cyan' },
}

function getToolMeta(toolName) {
  const key = (toolName || '').toLowerCase().replace(/-/g, '_')
  return TOOL_META[key] || { label: toolName || 'Tool', icon: 'wrench', color: 'cyan' }
}

/** Extract a short human-readable target from tool input JSON string */
function extractTarget(inputStr, toolName) {
  if (!inputStr) return ''
  try {
    const obj = JSON.parse(inputStr)
    // Common file path keys
    const path = obj.path || obj.file_path || obj.AbsolutePath || obj.file || obj.filename
      || obj.target_file || obj.TargetFile || obj.command || obj.query || ''
    if (!path) return inputStr.slice(0, 80)
    const short = String(path).split(/[\/\\]/).pop() || String(path)
    // If it has line numbers, add them
    const startLine = obj.start_line || obj.StartLine || obj.start
    const endLine = obj.end_line || obj.EndLine || obj.end
    if (startLine && endLine) return `${short} #L${startLine}-${endLine}`
    if (startLine) return `${short} #L${startLine}`
    return short
  } catch {
    return inputStr.slice(0, 80)
  }
}

function ToolActivityCard({ event }) {
  const [expanded, setExpanded] = useState(false)

  const isCall = event.type === 'tool_call'
  const isDone = event.done === true || event.type === 'tool_result'
  const toolName = event.tool || 'tool'
  const meta = getToolMeta(toolName)

  const target = isCall
    ? extractTarget(event.input, toolName)
    : (event.output || '')

  // Lines analyzed badge
  const lines = event.lines

  const iconMap = {
    eye:      <Eye size={12} />,
    pen:      <FilePen size={12} />,
    terminal: <Terminal size={12} />,
    folder:   <Folder size={12} />,
    search:   <Search size={12} />,
    wrench:   <Wrench size={12} />,
  }

  const colorMap = {
    cyan:  { accent: '#38bdf8', bg: 'rgba(56,189,248,0.06)',  border: 'rgba(56,189,248,0.18)' },
    green: { accent: '#10b981', bg: 'rgba(16,185,129,0.06)',  border: 'rgba(16,185,129,0.18)' },
    amber: { accent: '#f59e0b', bg: 'rgba(245,158,11,0.06)',  border: 'rgba(245,158,11,0.18)' },
    rose:  { accent: '#f43f5e', bg: 'rgba(244,63,94,0.06)',   border: 'rgba(244,63,94,0.18)'  },
    blue:  { accent: '#38bdf8', bg: 'rgba(2,132,199,0.06)',   border: 'rgba(2,132,199,0.18)'  },
  }

  const c = colorMap[meta.color] || colorMap.cyan
  const icon = iconMap[meta.icon] || iconMap.wrench

  const hasDetail = !isDone && event.input && event.input.length > 0

  return (
    <div
      className="tool-activity-card"
      style={{ borderColor: c.border, background: c.bg }}
      onClick={() => hasDetail && setExpanded(v => !v)}
    >
      <div className="tool-activity-header" style={{ cursor: hasDetail ? 'pointer' : 'default' }}>
        <span className="tool-activity-icon" style={{ color: c.accent }}>{icon}</span>
        <span className="tool-activity-label" style={{ color: c.accent }}>{meta.label}</span>
        {target && (
          <span className="tool-activity-target" title={event.input || event.output}>
            {target}
          </span>
        )}
        {lines != null && lines > 1 && (
          <span className="tool-activity-lines">({lines} lines)</span>
        )}
        <span className="tool-activity-status">
          {isDone
            ? <CheckCircle2 size={11} style={{ color: '#10b981' }} />
            : <RefreshCw size={11} className="spin-icon" style={{ color: c.accent }} />
          }
        </span>
        {hasDetail && (
          <span className="tool-activity-expand" style={{ color: c.accent }}>
            {expanded ? <ChevronDown size={11} /> : <ChevronRight size={11} />}
          </span>
        )}
      </div>
      {expanded && hasDetail && (
        <pre className="tool-activity-body">{event.input}</pre>
      )}
    </div>
  )
}

// ── Live Agent Thinking Card ─────────────────────────────────────────────────
function AgentThinkingCard() {
  return (
    <div className="agent-thinking-card">
      <div className="agent-thinking-avatar">
        <Sparkles size={12} />
      </div>
      <div className="agent-thinking-body">
        <div className="agent-thinking-dots">
          <span /><span /><span />
        </div>
        <span className="agent-thinking-label">Agent is working…</span>
      </div>
    </div>
  )
}

// --- Lightweight Markdown & Code Block Renderer ---
function CodeBlock({ code, lang }) {
  const [copied, setCopied] = useState(false)

  const handleCopy = () => {
    navigator.clipboard?.writeText(code)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  return (
    <div className="code-block-container">
      <div className="code-block-header">
        <span className="code-block-lang">{lang || 'code'}</span>
        <button className="code-copy-btn" onClick={handleCopy} title="Copy code">
          {copied ? <Check size={12} style={{ color: 'var(--accent-green)' }} /> : <Copy size={12} />}
          <span>{copied ? 'Copied' : 'Copy'}</span>
        </button>
      </div>
      <pre className="code-block-body">
        <code>{code}</code>
      </pre>
    </div>
  )
}

function FormattedContent({ content }) {
  if (!content) return null

  // Split on code fences ```lang\ncode```
  const parts = []
  const fenceRegex = /```([a-zA-Z0-9_-]*)\n([\s\S]*?)```/g
  let lastIndex = 0
  let match

  while ((match = fenceRegex.exec(content)) !== null) {
    if (match.index > lastIndex) {
      parts.push({ type: 'text', text: content.slice(lastIndex, match.index) })
    }
    parts.push({ type: 'code', lang: match[1] || 'text', code: match[2].trimEnd() })
    lastIndex = match.index + match[0].length
  }

  if (lastIndex < content.length) {
    parts.push({ type: 'text', text: content.slice(lastIndex) })
  }

  // Parse inline markdown elements for text chunks
  const renderTextChunk = (text, chunkIdx) => {
    const lines = text.split('\n')
    return (
      <div key={`chunk-${chunkIdx}`} className="markdown-text-chunk">
        {lines.map((line, lIdx) => {
          // Headers
          if (line.startsWith('### ')) {
            return <h4 key={lIdx} className="md-h4">{renderInline(line.slice(4))}</h4>
          }
          if (line.startsWith('## ')) {
            return <h3 key={lIdx} className="md-h3">{renderInline(line.slice(3))}</h3>
          }
          if (line.startsWith('# ')) {
            return <h2 key={lIdx} className="md-h2">{renderInline(line.slice(2))}</h2>
          }
          // Bullet points
          if (line.startsWith('- ') || line.startsWith('* ')) {
            return (
              <div key={lIdx} className="md-bullet">
                <span className="md-bullet-dot">•</span>
                <span className="md-bullet-text">{renderInline(line.slice(2))}</span>
              </div>
            )
          }
          // Empty line
          if (!line.trim()) {
            return <div key={lIdx} className="md-spacer" />
          }
          // Regular paragraph
          return <p key={lIdx} className="md-p">{renderInline(line)}</p>
        })}
      </div>
    )
  }

  const renderInline = (str) => {
    // Split on inline code `code`
    const codeTokens = str.split(/(`[^`]+`)/g)
    return codeTokens.map((token, tIdx) => {
      if (token.startsWith('`') && token.endsWith('`') && token.length > 2) {
        return <code key={tIdx} className="md-inline-code">{token.slice(1, -1)}</code>
      }
      // Bold **bold**
      const boldTokens = token.split(/(\*\*[^*]+\*\*)/g)
      return boldTokens.map((bToken, bIdx) => {
        if (bToken.startsWith('**') && bToken.endsWith('**') && bToken.length > 4) {
          return <strong key={`${tIdx}-${bIdx}`}>{bToken.slice(2, -2)}</strong>
        }
        return bToken
      })
    })
  }

  return (
    <div className="formatted-message">
      {parts.map((p, idx) => {
        if (p.type === 'code') {
          return <CodeBlock key={idx} lang={p.lang} code={p.code} />
        }
        return renderTextChunk(p.text, idx)
      })}
    </div>
  )
}

export default function App() {
  const [sessions, setSessions] = useState([])
  const [activeSessionId, setActiveSessionId] = useState('')
  const [messages, setMessages] = useState([])
  const [prompt, setPrompt] = useState('')
  const [isRunning, setIsRunning] = useState(false)
  const [pendingApproval, setPendingApproval] = useState(null)
  const [attachments, setAttachments] = useState([])
  const [selectedModel, setSelectedModel] = useState('groq:qwen/qwen3.8-27b')
  const [isOnline, setIsOnline] = useState(false)
  const [activeDrawer, setActiveDrawer] = useState(null) // null | 'sessions' | 'workspace'
  
  const fileInputRef = useRef(null)
  const textareaRef = useRef(null)
  const chatEndRef = useRef(null)

  // Workspace explorer states
  const [rightPanelTab, setRightPanelTab] = useState('files') // 'files' | 'artifacts' | 'guard'
  const [files, setFiles] = useState([])
  const [selectedFile, setSelectedFile] = useState(null)
  const [fileContent, setFileContent] = useState('')
  const [workspaceInfo, setWorkspaceInfo] = useState(null)
  const [expandedFolders, setExpandedFolders] = useState({})
  const [folderContents, setFolderContents] = useState({})

  // Connect helper to wake VS Code extension API
  const handleConnect = () => {
    try {
      if (typeof window !== 'undefined' && window.acquireVsCodeApi) {
        window.vscodeApi = window.vscodeApi || window.acquireVsCodeApi()
        window.vscodeApi?.postMessage({ command: 'startApi' })
      }
    } catch (_) {}
    fetchWorkspaceInfo()
    fetchSessions()
  }

  // Initial load and polling
  useEffect(() => {
    fetchSessions()
    fetchFiles()
    fetchWorkspaceInfo()
    const interval = setInterval(() => {
      fetchSessions()
      fetchFiles()
      fetchWorkspaceInfo()
    }, 3000)
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
  }, [messages, pendingApproval, isRunning])

  // --- Session Management ---
  const fetchWorkspaceInfo = async () => {
    try {
      const res = await fetch(`${API_BASE}/workspace`)
      if (res.ok) {
        setIsOnline(true)
        const data = await res.json()
        setWorkspaceInfo(prev => {
          if (prev?.workspace_root === data.workspace_root) return prev
          return data
        })
      } else {
        setIsOnline(false)
      }
    } catch (_) {
      setIsOnline(false)
    }
  }

  const fetchSessions = async () => {
    try {
      const res = await fetch(`${API_BASE}/sessions`)
      if (res.ok) {
        setIsOnline(true)
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
    } catch (_) {
      setIsOnline(false)
    }
  }

  const handleSelectSession = (sessionId) => {
    if (sessionId === activeSessionId) return
    setMessages([])
    setPendingApproval(null)
    setAttachments([])
    setActiveSessionId(sessionId)
    setActiveDrawer(null)
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
        setMessages([])
        setPendingApproval(null)
        setAttachments([])
        setActiveSessionId(newSession.id)
        setActiveDrawer(null)
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
          handleSelectSession(remaining[0].id)
        } else {
          handleCreateSession()
        }
      }
    } catch (err) {
      console.error(err)
    }
  }

  const handleClearChat = () => {
    setMessages([])
    setPendingApproval(null)
    setAttachments([])
  }

  const loadSessionState = async (threadId) => {
    try {
      const res = await fetch(`${API_BASE}/runs/${threadId}`)
      if (res.ok) {
        const state = await res.json()
        if (state && state.messages && state.messages.length > 0) {
          const mapped = state.messages.map((m, idx) => {
            // Determine sender
            const sender = m.sender
              || (m.type === 'human' || m.role === 'user' ? 'user' : 'agent')

            // Detect tool call / tool result messages and convert to [TOOL] markup
            const isToolMsg = m.type === 'tool' || m.type === 'tool_result'
              || m.type === 'function_call' || m.type === 'tool_call'
              || m.role === 'tool'

            let text = ''
            if (isToolMsg) {
              const toolName = m.tool_name || m.name || m.type || 'tool'
              const toolInput = m.tool_input || m.input || m.arguments || ''
              const toolOutput = m.tool_output || m.output || m.content || ''
              const inputStr = typeof toolInput === 'object'
                ? JSON.stringify(toolInput, null, 2)
                : String(toolInput || '')
              const outputStr = typeof toolOutput === 'object'
                ? JSON.stringify(toolOutput, null, 2)
                : String(toolOutput || '')
              // Only show the summary target (first meaningful line), not full file content
              const summaryLine = inputStr.split('\n')[0].replace(/['"{}]/g, '').trim().slice(0, 100)
              text = `[TOOL:${toolName}]${summaryLine}[/TOOL]`
              // Append brief output summary if it's not a massive file dump
              if (outputStr && outputStr.length < 500) {
                text += `\n${outputStr}`
              } else if (outputStr) {
                text += `\n_(${outputStr.split('\n').length} lines analyzed)_`
              }
            } else {
              text = typeof m.text === 'string'
                ? m.text
                : (typeof m.content === 'string'
                  ? m.content
                  : JSON.stringify(m.content || m))
            }

            return {
              id: m.id || `msg-${idx}`,
              sender,
              time: m.time || new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
              text,
              attachments: m.attachments || [],
            }
          })
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
    } catch (_) {}
  }

  // --- File Attachments ---
  const handleFileSelect = (e) => {
    const fileList = Array.from(e.target.files || [])
    if (!fileList.length) return

    fileList.forEach(file => {
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

    if (e.target) e.target.value = ''
  }

  const handleRemoveAttachment = (id) => {
    setAttachments(prev => prev.filter(a => a.id !== id))
  }

  const runAbortControllerRef = useRef(null)

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
      console.warn('Stop run error', e)
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

  const handleSendMessage = async (customPrompt) => {
    const messageToSend = typeof customPrompt === 'string' ? customPrompt : prompt
    if (isRunning) {
      handleStopRun()
      return
    }
    if (!messageToSend.trim() && attachments.length === 0) return

    const rawPrompt = messageToSend
    const currentAttachments = [...attachments]
    setPrompt('')
    setAttachments([])

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

    // Create a live agent message that accumulates tool cards + final text
    const agentMsgId = `agent-${Date.now()}`
    const agentMsgTime = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })

    // toolEvents: array of {type:'tool_call'|'tool_result', tool, input?, output?, id, lines?}
    // finalText: the final prose response
    // We store these directly in the message as structured data
    setMessages(prev => [...prev, {
      id: agentMsgId,
      sender: 'agent',
      time: agentMsgTime,
      text: '',
      toolEvents: [],
      streaming: true,
    }])

    try {
      const res = await fetch(`${API_BASE}/runs/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        signal: controller.signal,
        body: JSON.stringify({
          user_request: finalPrompt,
          thread_id: activeSessionId,
          model: selectedModel,
        }),
      })

      if (!res.ok) {
        throw new Error(`HTTP ${res.status}`)
      }

      const reader = res.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ''

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n')
        buffer = lines.pop() // keep incomplete line

        for (const line of lines) {
          if (!line.startsWith('data: ')) continue
          const jsonStr = line.slice(6).trim()
          if (!jsonStr) continue

          let evt
          try { evt = JSON.parse(jsonStr) } catch { continue }

          if (evt.type === 'tool_call') {
            setMessages(prev => prev.map(m => m.id === agentMsgId
              ? { ...m, toolEvents: [...(m.toolEvents || []), { ...evt, done: false }] }
              : m
            ))
          } else if (evt.type === 'tool_result') {
            setMessages(prev => prev.map(m => {
              if (m.id !== agentMsgId) return m
              const events = [...(m.toolEvents || [])]
              // Mark matching tool_call as done and attach result summary
              const callIdx = events.findLastIndex?.(e => e.type === 'tool_call' && e.id === evt.id && !e.done)
              if (callIdx !== undefined && callIdx >= 0) {
                events[callIdx] = { ...events[callIdx], done: true, resultSummary: evt.output, lines: evt.lines }
              } else {
                events.push({ ...evt, done: true })
              }
              return { ...m, toolEvents: events }
            }))
          } else if (evt.type === 'text_delta') {
            setMessages(prev => prev.map(m => m.id === agentMsgId
              ? { ...m, text: (m.text || '') + evt.text }
              : m
            ))
          } else if (evt.type === 'done') {
            setIsRunning(false)
            setMessages(prev => prev.map(m => m.id === agentMsgId
              ? { ...m, text: evt.output || m.text || 'Task completed.', streaming: false }
              : m
            ))
            if (evt.status === 'interrupted' || evt.ask_approval) {
              setPendingApproval({
                threadId: activeSessionId,
                description: evt.description || 'Action triggered an "ask" rule in permissions.json',
                action: evt.action || 'Shell / File modification',
              })
            }
            fetchFiles()
          }
        }
      }

      // Ensure streaming flag is cleared
      setIsRunning(false)
      setMessages(prev => prev.map(m => m.id === agentMsgId ? { ...m, streaming: false } : m))
      fetchFiles()

    } catch (err) {
      setIsRunning(false)
      setMessages(prev => prev.map(m => m.id === agentMsgId ? { ...m, streaming: false } : m))
      if (err.name === 'AbortError') return

      const isFetchFail = err.message && err.message.toLowerCase().includes('fetch')
      if (isFetchFail) {
        try {
          if (typeof window !== 'undefined' && window.acquireVsCodeApi) {
            window.vscodeApi = window.vscodeApi || window.acquireVsCodeApi();
            window.vscodeApi?.postMessage({ command: 'startApi' });
          }
        } catch (_) {}
      }
      const errorText = isFetchFail
        ? `⚠️ **Backend Server Offline:** Could not reach the API at \`${API_BASE}\`.\n\n` +
          `Starting the server automatically now… Please wait a few seconds and try sending again!\n` +
          `*(If needed, run manually: \`Ctrl+Shift+P\` → \`AI Coding Agent: Start API Server\`)*`
        : `⚠️ **Request Execution Error:** ${err.message}`

      setMessages(prev => prev.map(m => m.id === agentMsgId
        ? { ...m, text: errorText, streaming: false }
        : m
      ))
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
          text: approved
            ? `✅ **Action Approved.** Agent resumed execution.\n\n${JSON.stringify(data, null, 2)}`
            : `❌ **Action Rejected by User.** Agent stopped safely.`,
        }
      ])
      fetchFiles()
    } catch (err) {
      setIsRunning(false)
      console.error(err)
    }
  }

  // --- Workspace File Explorer ---
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
    } catch (_) {}
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
            style={{ paddingLeft: `${8 + depth * 12}px` }}
            onClick={() => handleSelectFile(f)}
          >
            {isDir ? (
              <>
                <span className="file-tree-arrow">
                  {isExpanded ? <ChevronDown size={11} /> : <ChevronRight size={11} />}
                </span>
                <Folder size={13} style={{ color: 'var(--accent-amber)', flexShrink: 0 }} />
              </>
            ) : (
              <>
                <span style={{ width: '11px', display: 'inline-block' }} />
                <FileCode size={13} style={{ color: 'var(--accent-cyan)', flexShrink: 0 }} />
              </>
            )}
            <span className="file-tree-name">{f.name}</span>
          </div>

          {isDir && isExpanded && (
            <div className="file-tree-subfolder">
              {children.length === 0 ? (
                <div style={{ paddingLeft: `${24 + depth * 12}px`, fontSize: '11px', color: 'var(--text-dim)', fontStyle: 'italic', paddingBottom: '2px' }}>
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

  const quickPrompts = [
    { label: '⚡ Explain Project', prompt: 'Explain the project architecture and what each folder is responsible for.' },
    { label: '🔍 Find Bugs', prompt: 'Inspect the codebase for any syntax bugs, race conditions, or unhandled exceptions.' },
    { label: '🧪 Generate Tests', prompt: 'Generate unit tests for the core modules and show how to run them.' },
    { label: '🛠️ Refactor Code', prompt: 'Recommend key refactoring improvements for code readability and modularity.' },
  ]

  return (
    <div className="app-container">
      {/* Top ACP Header */}
      <header className="acp-header">
        <div className="acp-header-top">
          <div className="acp-brand">
            <div className="acp-brand-icon">
              <Sparkles size={16} />
            </div>
            <span>Coding Agent</span>
          </div>

          <div className="acp-header-actions">
            <button
              className="acp-icon-btn"
              onClick={() => handleCreateSession()}
              title="New Chat Session"
            >
              <Plus size={15} />
            </button>
            <button
              className={`acp-icon-btn ${activeDrawer === 'sessions' ? 'active' : ''}`}
              onClick={() => setActiveDrawer(activeDrawer === 'sessions' ? null : 'sessions')}
              title="Chat History"
            >
              <History size={14} />
            </button>
            <button
              className={`acp-icon-btn ${activeDrawer === 'workspace' ? 'active' : ''}`}
              onClick={() => setActiveDrawer(activeDrawer === 'workspace' ? null : 'workspace')}
              title="Workspace Files"
            >
              <FolderTree size={14} />
            </button>
            <button
              className="acp-icon-btn"
              onClick={handleClearChat}
              title="Clear Current Messages"
            >
              <Trash2 size={14} />
            </button>
          </div>
        </div>

        <div className="acp-header-meta">
          <div className="acp-status-pill">
            <span className={`status-dot ${isOnline ? 'online' : 'offline'}`} />
            <span>{isOnline ? 'Backend Connected' : 'Offline'}</span>
            {!isOnline && (
              <button className="btn-reconnect" onClick={handleConnect} title="Retry backend connection">
                Retry
              </button>
            )}
          </div>
        </div>
      </header>

      {/* ── Slide-over Drawers (Sessions History & Workspace Files) ── */}
      {activeDrawer === 'sessions' && (
        <div className="drawer-panel">
          <div className="drawer-header">
            <div className="drawer-title">
              <History size={14} />
              <span>Chat Threads</span>
            </div>
            <button className="drawer-close-btn" onClick={() => setActiveDrawer(null)}>
              <X size={14} />
            </button>
          </div>
          <div className="drawer-content">
            <button className="drawer-new-btn" onClick={() => handleCreateSession()}>
              <Plus size={14} />
              <span>New Thread</span>
            </button>
            <div className="drawer-list">
              {sessions.map(s => (
                <div
                  key={s.id}
                  className={`drawer-item ${activeSessionId === s.id ? 'active' : ''}`}
                  onClick={() => handleSelectSession(s.id)}
                >
                  <div className="drawer-item-title">
                    <Code2 size={13} />
                    <span>{s.title || `Thread ${s.id.slice(0, 8)}`}</span>
                  </div>
                  <button
                    className="drawer-item-del"
                    onClick={(e) => handleDeleteSession(s.id, e)}
                    title="Delete thread"
                  >
                    <Trash2 size={12} />
                  </button>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {activeDrawer === 'workspace' && (
        <div className="drawer-panel">
          <div className="drawer-header">
            <div className="drawer-title">
              <FolderTree size={14} />
              <span>{workspaceInfo?.workspace_name || 'Workspace'}</span>
            </div>
            <div style={{ display: 'flex', gap: '6px' }}>
              <button className="acp-icon-btn" onClick={() => fetchFiles()} title="Refresh">
                <RefreshCw size={12} />
              </button>
              <button className="drawer-close-btn" onClick={() => setActiveDrawer(null)}>
                <X size={14} />
              </button>
            </div>
          </div>
          <div className="drawer-content">
            {!selectedFile ? (
              <div className="workspace-tree-view">
                {renderFileTree(files)}
              </div>
            ) : (
              <div className="code-viewer-container">
                <div className="code-viewer-header">
                  <span className="code-viewer-path">{selectedFile.path}</span>
                  <button className="code-viewer-close" onClick={() => setSelectedFile(null)}>
                    Back to files
                  </button>
                </div>
                <pre className="code-viewer-body">{fileContent}</pre>
              </div>
            )}
          </div>
        </div>
      )}

      {/* ── Main Chat Stream ── */}
      <main className="chat-main">
        {messages.length === 0 ? (
          <div className="welcome-hero">
            <div className="welcome-icon-wrap">
              <Sparkles size={28} />
            </div>
            <h2 className="welcome-title">Welcome to Coding Agent</h2>
            <p className="welcome-desc">
              Autonomous AI pair programmer right inside your VS Code.
            </p>

            {workspaceInfo && (
              <div className="welcome-workspace-pill">
                <Folder size={12} />
                <span>{workspaceInfo.workspace_name}</span>
              </div>
            )}

            <div className="quick-prompts-grid">
              {quickPrompts.map((qp, idx) => (
                <button
                  key={idx}
                  className="quick-prompt-btn"
                  onClick={() => handleSendMessage(qp.prompt)}
                >
                  {qp.label}
                </button>
              ))}
            </div>
          </div>
        ) : (
          <div className="chat-scroll">
            {messages.map(msg => {
              const parts = msg.sender === 'agent' ? parseMessageParts(msg.text) : null
              const hasToolParts = parts && parts.some(p => p.type === 'tool')

              return (
                <div key={msg.id} className={`message-item ${msg.sender === 'user' ? 'from-user' : 'from-agent'}`}>
                  <div className={`message-avatar ${msg.sender === 'user' ? 'avatar-user' : 'avatar-agent'}`}>
                    {msg.sender === 'user' ? 'U' : <Sparkles size={14} />}
                  </div>

                  <div className="message-content-wrapper">
                    <div className="message-meta">
                      <span className="message-sender">{msg.sender === 'user' ? 'You' : 'Coding Agent'}</span>
                      <span className="message-time">{msg.time}</span>
                    </div>

                    {/* User messages: plain bubble */}
                    {msg.sender === 'user' && (
                      <div className="message-bubble">
                        {msg.attachments && msg.attachments.length > 0 && (
                          <div className="message-attachments-preview">
                            {msg.attachments.map(att => (
                              <div key={att.id} className="message-att-item">
                                {att.type === 'image' ? (
                                  <img src={att.dataUrl} alt={att.name} className="message-att-thumb" />
                                ) : (
                                  <File size={12} />
                                )}
                                <span className="message-att-name">{att.name}</span>
                              </div>
                            ))}
                          </div>
                        )}
                        <FormattedContent content={msg.text} />
                      </div>
                    )}

                    {/* Agent messages: live tool event cards + prose */}
                    {msg.sender === 'agent' && (
                      <div className="agent-message-parts">
                        {/* Live SSE tool events */}
                        {(msg.toolEvents || []).map((evt, ei) => (
                          <ToolActivityCard key={`${evt.id || ei}-${evt.type}`} event={evt} />
                        ))}
                        {/* Streaming thinking indicator */}
                        {msg.streaming && (msg.toolEvents || []).length === 0 && (
                          <AgentThinkingCard />
                        )}
                        {/* Final prose response */}
                        {msg.text && (
                          <div className="message-bubble">
                            <FormattedContent content={msg.text} />
                          </div>
                        )}
                        {/* Fallback: old-style [TOOL:] text parsing */}
                        {!msg.toolEvents && !msg.streaming && !msg.text && (
                          <div className="message-bubble">
                            <FormattedContent content={msg.text} />
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                </div>
              )
            })}

            {/* Pending HITL Approval Gate Card */}
            {pendingApproval && (
              <div className="guard-card">
                <div className="guard-header">
                  <ShieldAlert size={16} />
                  <span>ACTION APPROVAL REQUIRED</span>
                </div>
                <div className="guard-desc">
                  {pendingApproval.description}
                  <div className="guard-action-preview">
                    {pendingApproval.action}
                  </div>
                </div>
                <div className="guard-actions">
                  <button className="btn-approve" onClick={() => handleApproval(true)}>
                    <CheckCircle2 size={13} />
                    <span>Approve & Continue</span>
                  </button>
                  <button className="btn-reject" onClick={() => handleApproval(false)}>
                    <XCircle size={13} />
                    <span>Reject</span>
                  </button>
                </div>
              </div>
            )}

            {isRunning && <AgentThinkingCard />}

            <div ref={chatEndRef} />
          </div>
        )}

        {/* ── Fixed Bottom Prompt Dock (Antigravity & Codex style) ── */}
        <div className="input-dock">
          <form className="input-box" onSubmit={(e) => { e.preventDefault(); handleSendMessage(); }}>
            {attachments.length > 0 && (
              <div className="attachments-tray">
                {attachments.map(att => (
                  <div key={att.id} className="attachment-chip">
                    {att.type === 'image' ? (
                      <img src={att.dataUrl} alt={att.name} className="attachment-chip-thumb" />
                    ) : (
                      <File size={12} />
                    )}
                    <span className="attachment-chip-name">{att.name}</span>
                    <button
                      type="button"
                      className="attachment-remove-btn"
                      onClick={() => handleRemoveAttachment(att.id)}
                    >
                      <X size={11} />
                    </button>
                  </div>
                ))}
              </div>
            )}

            <textarea
              ref={textareaRef}
              className="prompt-textarea"
              placeholder="Ask your agent (e.g. 'Build an API', 'Fix test failure')..."
              value={prompt}
              rows={2}
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
              <div className="input-controls-left">
                <button
                  type="button"
                  className="attach-btn"
                  onClick={() => fileInputRef.current?.click()}
                  title="Attach file or screenshot (+)"
                >
                  <Plus size={15} />
                </button>

                <div className="dock-model-selector" title="Choose AI Model">
                  <Cpu size={12} className="model-icon" />
                  <select
                    className="dock-model-dropdown"
                    value={selectedModel}
                    onChange={(e) => setSelectedModel(e.target.value)}
                  >
                    <option value="groq:qwen/qwen3.8-27b">qwen/qwen3.8-27b (Groq)</option>
                    <option value="ollama:gemma4:cloud">gemma4:cloud</option>
                    <option value="ollama:nvidia-nemotron-3-super:cloud">nvidia-nemotron-3-super:cloud</option>
                    <option value="ollama:gpt-oss:120b-cloud">gpt-oss-120b:cloud</option>
                  </select>
                </div>
              </div>

              <div className="input-controls-right">
                {workspaceInfo && (
                  <span className="workspace-badge" title={workspaceInfo.workspace_root}>
                    <Folder size={11} />
                    <span>{workspaceInfo.workspace_name}</span>
                  </span>
                )}

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
                    <Play size={13} fill="currentColor" />
                  )}
                  <span className="send-btn-label">{isRunning ? 'Stop' : 'Send'}</span>
                </button>
              </div>
            </div>
          </form>
        </div>
      </main>
    </div>
  )
}
