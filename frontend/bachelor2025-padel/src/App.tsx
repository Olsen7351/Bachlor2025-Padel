import {Route, Routes} from "react-router-dom"
import './App.css'
import LoginPage from "./features/login/LoginPage.tsx";
import UploadPage from "./features/upload/UploadPage.tsx";
import AnalysisDashboard from "./features/analysis/AnalysisDashboard.tsx";
import DuelsPage from "./features/analysis/DuelsPage.tsx";

function App() {

  return (
              <Routes>
                  <Route index path="/" element={<LoginPage />} />
                  <Route path="/upload" element={<UploadPage/>} />
                  <Route path="/matches/:videoId/overview" element={<AnalysisDashboard/>} />
                  <Route path="/matches/:videoId/overview/duels" element={<DuelsPage/>} />
              </Routes>
  )
}

export default App
