import {Route, Routes} from "react-router-dom"
import './App.css'
import LoginPage from "./features/login/LoginPage.tsx";
import UploadPage from "./features/upload/UploadPage.tsx";
import AnalysisDashboard from "./features/analysis/AnalysisDashboard.tsx";
import DuelsPage from "./features/analysis/DuelsPage.tsx";
import HitsPage from "./features/analysis/HitsPage.tsx";
import HeatMapPage from "./features/analysis/HeatMapPage.tsx";
import RegisterPage from "./features/login/RegisterPage.tsx";
import DashboardPage from "./features/dashboard/DashboardPage.tsx";
import ProtectedRoute from "./utils/ProtectedRoute.tsx";

function App() {

  return (
              <Routes>
                  <Route index path="/" element={<LoginPage />} />
                  <Route path="/register" element={<RegisterPage/>} />

                  <Route element={<ProtectedRoute />}>
                  <Route path="/dashboard" element={<DashboardPage/>} />
                  <Route path="/upload" element={<UploadPage/>} />
                  <Route path="/matches/:videoId/overview" element={<AnalysisDashboard/>} />
                  <Route path="/matches/:videoId/overview/duels" element={<DuelsPage/>} />
                  <Route path="/matches/:videoId/overview/hits" element={<HitsPage/>} />
                  <Route path="/matches/:videoId/overview/heatmap" element={<HeatMapPage/>} />
                    </Route>

              </Routes>
  )
}

export default App
