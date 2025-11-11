import React, { useState } from 'react'
import axios from 'axios'
import { useNavigate } from 'react-router-dom'
const API = import.meta.env.VITE_API_URL || 'http://localhost:5000'

export default function Login(){
  const [identifier, setIdentifier] = useState('')
  const [password, setPassword] = useState('')
  const nav = useNavigate()

  async function submit(e){
    e.preventDefault()
    try{
      const res = await axios.post(`${API}/api/login`, { username: identifier, password })
      localStorage.setItem('user', JSON.stringify(res.data.user))
      nav('/')
    }catch(e){ alert('Login failed') }
  }
  return (
    <div className="max-w-md mx-auto bg-[#041025] p-6 rounded mt-8">
      <h2 className="text-xl font-semibold">Login</h2>
      <form onSubmit={submit} className="space-y-3 mt-4">
        <input value={identifier} onChange={e=>setIdentifier(e.target.value)} placeholder="email or username" className="w-full p-2 rounded bg-[#061526]" />
        <input type="password" value={password} onChange={e=>setPassword(e.target.value)} placeholder="password" className="w-full p-2 rounded bg-[#061526]" />
        <button className="w-full bg-indigo-600 p-2 rounded">Login</button>
      </form>
    </div>
  )
}
