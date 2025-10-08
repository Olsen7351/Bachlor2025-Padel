import {Route, Routes } from "react-router-dom"
import './App.css'
import LoginPage from "./features/login/LoginPage.tsx";
import UploadPage from "./features/login/UploadPage.tsx";

function App() {

  return (
              <Routes>
                  <Route index path="/" element={<LoginPage />} />
                  <Route path="/upload" element={<UploadPage/>} />
              </Routes>
  )
}

export default App
