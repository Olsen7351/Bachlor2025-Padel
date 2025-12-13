import {BrowserRouter, Route, Routes} from "react-router-dom"
import './App.css'
import LoginPage from "./features/login/LoginPage.tsx";
import UploadPage from "./features/upload/UploadPage.tsx";
import AnalysisDashboard from "./features/analysis/AnalysisDashboard.tsx";

function App() {

  return (
      <BrowserRouter>
              <Routes>
                  <Route index path="/" element={<LoginPage />} />
                  <Route path="/upload" element={<UploadPage/>} />
                  <Route path="/id" element={<AnalysisDashboard/>} />
              </Routes>
      </BrowserRouter>

  )
}

export default App
