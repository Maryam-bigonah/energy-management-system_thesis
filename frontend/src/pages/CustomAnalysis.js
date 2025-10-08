import React from 'react';
import { Box, Typography } from '@mui/material';
import CustomAnalysisForm from '../components/CustomAnalysisForm';

const CustomAnalysis = () => {
  return (
    <Box>
      <Typography variant="h3" component="h1" gutterBottom sx={{ fontWeight: 'bold', mb: 4 }}>
        🔧 Custom Solar Analysis
      </Typography>
      
      <Typography variant="body1" color="text.secondary" sx={{ mb: 4, maxWidth: 800 }}>
        Configure your own parameters for PV capacity estimation and PVGIS simulation. 
        This tool allows you to customize all aspects of the solar analysis based on your specific requirements.
      </Typography>

      <CustomAnalysisForm />
    </Box>
  );
};

export default CustomAnalysis;
