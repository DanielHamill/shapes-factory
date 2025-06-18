import './App.css';
import * as React from "react";
import DrawingCanvas from './components/DrawingCanvas';
import { Routes, Route } from 'react-router-dom';

// function Home() {
//   return (
//     <div>
//       <h1>AI Shapes Factory</h1>
//       <DrawingCanvas />
//     </div>
//   );
// }

// function App() {
//   return (
//     <Routes>
//         <Route exact path="/" element={<Home />} />
//     </Routes>
//   );
// }


function App() {
  return (
    <div>
       <h1>AI Shapes Factory</h1>
       <DrawingCanvas />
    </div>
  );
}

export default App;
