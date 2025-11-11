import React, { useState } from 'react'
import axios from 'axios'
import { useNavigate } from 'react-router-dom'
const API = import.meta.env.VITE_API_URL || 'http://localhost:5000'

export default function Signup(){
  const [username,setUsername]=useState('')
  const [email,setEmail]=useState('')
  const [password,setPassword]=useState('')
  const nav=useNavigate()

  async function submit(e){
    e.preventDefault()
    try{
      const res = await axios.post(`${API}/api/signup`, { username, email, password })
      localStorage.setItem('user', JSON.stringify(res.data.user))
      nav('/')
    }catch(e){ alert('Signup failed') }
  }

  return (
    <div className="max-w-md mx-auto bg-[#041025] p-6 rounded mt-8">
      <h2 className="text-xl font-semibold">Sign up</h2>
      <form onSubmit={submit} className="space-y-3 mt-4">
        <input value={username} onChange={e=>setUsername(e.target.value)} placeholder="username" className="w-full p-2 rounded bg-[#061526]" />
        <input value={email} onChange={e=>setEmail(e.target.value)} placeholder="email" className="w-full p-2 rounded bg-[#061526]" />
        <input type="password" value={password} onChange={e=>setPassword(e.target.value)} placeholder="password" className="w-full p-2 rounded bg-[#061526]" />
        <button className="w-full bg-indigo-600 p-2 rounded">Create account</button>
      </form>
    </div>
  )
}
