import React from 'react'
export default function Dashboard(){
  const user = JSON.parse(localStorage.getItem('user')|| "null")
  return (
    <div className="glass p-6 rounded-2xl">
      <h2 className="text-xl font-semibold">Dashboard</h2>
      <p className="text-gray-400 mt-2">Welcome {user?.username || "guest"}. Saved items appear on the Home page in Recent.</p>
    </div>
  )
}
