import React from 'react'
import { Routes, Route } from 'react-router-dom'
import Navbar from './components/Navbar'
import UploadPanel from './components/UploadPanel'
import Login from './components/Login'
import Signup from './components/Signup'
import Dashboard from './components/Dashboard'

export default function App(){
  return (
    <div>
      <Navbar/>
      <main className="max-w-6xl mx-auto p-6">
        <Routes>
          <Route path="/" element={<UploadPanel />} />
          <Route path="/dashboard" element={<Dashboard/>} />
          <Route path="/login" element={<Login/>} />
          <Route path="/signup" element={<Signup/>} />
        </Routes>
      </main>
    </div>
  )
}
