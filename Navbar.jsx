import React from 'react'
import { Link, useNavigate } from 'react-router-dom'

export default function Navbar(){
  const navigate = useNavigate()
  const user = JSON.parse(localStorage.getItem('user') || "null")
  const logout = ()=>{ localStorage.removeItem('user'); navigate('/login') }

  return (
    <header className="bg-[#041025] border-b border-gray-800">
      <div className="max-w-6xl mx-auto px-4 py-3 flex items-center justify-between">
        <Link to="/" className="font-bold text-2xl h1-neon">AI • STT</Link>
        <nav className="flex items-center gap-4">
          <Link to="/" className="text-gray-300">Home</Link>
          <Link to="/dashboard" className="text-gray-300">Dashboard</Link>
          {user ? (
            <>
              <span className="text-gray-400">Hi, {user.username}</span>
              <button onClick={logout} className="bg-red-500 text-white px-3 py-1 rounded">Logout</button>
            </>
          ) : (
            <>
              <Link to="/login" className="text-gray-300">Login</Link>
              <Link to="/signup" className="bg-indigo-600 text-white px-3 py-1 rounded">Sign up</Link>
            </>
          )}
        </nav>
      </div>
    </header>
  )
}

