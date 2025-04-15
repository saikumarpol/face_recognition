import React, { useState, useRef, useEffect } from 'react';
import {
  Box,
  Button,
  Typography,
  TextField,
  Grid,
  Alert,
  CircularProgress,
  Paper,
} from '@mui/material';
import { Camera } from 'lucide-react';

const FaceDetectionSystem = () => {
  const videoRef = useRef(null);
  const [stream, setStream] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [recognizedPerson, setRecognizedPerson] = useState(null);
  const [showForm, setShowForm] = useState(false);
  const [formData, setFormData] = useState({
    firstName: '',
    lastName: '',
    age: '',
    height: '',
    weight: '',
    capturedImage: null,
  });

  useEffect(() => {
    startCamera();
    return () => {
      if (stream) {
        stream.getTracks().forEach(track => track.stop());
      }
    };
  }, []);

  const startCamera = async () => {
    try {
      const streamData = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480 },
      });
      setStream(streamData);
      if (videoRef.current) {
        videoRef.current.srcObject = streamData;
      }
    } catch (err) {
      setError('Error accessing webcam: ' + err.message);
    }
  };

  const captureAndRecognize = async () => {
    setLoading(true);
    setError(null);
    setRecognizedPerson(null);
    setShowForm(false);

    try {
      const canvas = document.createElement('canvas');
      canvas.width = videoRef.current.videoWidth;
      canvas.height = videoRef.current.videoHeight;
      const ctx = canvas.getContext('2d');
      ctx.drawImage(videoRef.current, 0, 0);

      const blob = await new Promise(resolve =>
        canvas.toBlob(resolve, 'image/jpeg', 0.95)
      );
      const fd = new FormData();
      fd.append('image', blob, 'capture.jpg');

      const response = await fetch('http://localhost:8000/recognize_or_add/', {
        method: 'POST',
        body: fd,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Recognition failed');
      }

      const data = await response.json();

      if (data.status === 'recognized') {
        setRecognizedPerson(data.person);
      } else if (data.status === 'not_found') {
        setShowForm(true);
        setFormData(prev => ({ ...prev, capturedImage: blob }));
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleInputChange = e => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value,
    });
  };

  const handleSubmit = async e => {
    e.preventDefault();
    setLoading(true);
    setError(null);

    try {
      if (!formData.firstName.trim() || !formData.lastName.trim()) {
        throw new Error('First name and last name are required');
      }

      const submitFormData = new FormData();
      submitFormData.append('image', formData.capturedImage, 'capture.jpg');
      submitFormData.append('first_name', formData.firstName.trim());
      submitFormData.append('last_name', formData.lastName.trim());
      if (formData.age) submitFormData.append('age', formData.age.trim());
      if (formData.height) submitFormData.append('height', formData.height.trim());
      if (formData.weight) submitFormData.append('weight', formData.weight.trim());

      const response = await fetch('http://localhost:8000/add_person/', {
        method: 'POST',
        body: submitFormData,
      });

      const data = await response.json();

      if (!response.ok) {
        if (data.detail?.includes('already registered') && data.existing_person) {
          setShowForm(false);
          setRecognizedPerson({
            ...data.existing_person,
            confidence: data.confidence || 'Unknown',
            message: 'Face already registered in system',
          });
          return;
        }
        throw new Error(data.detail || 'Failed to add person');
      }

      setShowForm(false);
      setRecognizedPerson({
        first_name: formData.firstName,
        last_name: formData.lastName,
        message: 'Successfully added to the system',
        person_id: data.person_id,
      });

      setFormData({
        firstName: '',
        lastName: '',
        age: '',
        height: '',
        weight: '',
        capturedImage: null,
      });
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <Paper elevation={4} sx={{ maxWidth: 800, mx: 'auto', mt: 4, p: 4 }}>
      <Typography variant="h4" gutterBottom fontWeight="bold">
        Face Recognition System
      </Typography>

      <Box sx={{ position: 'relative', mb: 2 }}>
        <video
          ref={videoRef}
          autoPlay
          playsInline
          style={{ width: '100%', borderRadius: '8px' }}
        />
        {loading && (
          <Box
            sx={{
              position: 'absolute',
              top: 0,
              bottom: 0,
              left: 0,
              right: 0,
              bgcolor: 'rgba(0,0,0,0.4)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              borderRadius: '8px',
            }}
          >
            <CircularProgress color="inherit" />
          </Box>
        )}
      </Box>

      <Button
        fullWidth
        onClick={captureAndRecognize}
        variant="contained"
        color="primary"
        startIcon={<Camera size={20} />}
        disabled={loading}
        sx={{ mb: 2 }}
      >
        {loading ? 'Processing...' : 'Detect Face'}
      </Button>

      {error && (
        <Alert severity="error" sx={{ mb: 2 }}>
          {error}
        </Alert>
      )}

      {recognizedPerson && (
        <Alert severity="success" sx={{ mb: 2 }}>
          <Typography variant="subtitle1" fontWeight="bold">
            Person Recognized:
          </Typography>
          <Typography>
            Name: {recognizedPerson.first_name} {recognizedPerson.last_name}
          </Typography>
          {recognizedPerson.age && <Typography>Age: {recognizedPerson.age}</Typography>}
          {recognizedPerson.height && <Typography>Height: {recognizedPerson.height}</Typography>}
          {recognizedPerson.weight && <Typography>Weight: {recognizedPerson.weight}</Typography>}
          {recognizedPerson.confidence && (
            <Typography>Confidence: {recognizedPerson.confidence}</Typography>
          )}
          {recognizedPerson.message && (
            <Typography sx={{ fontWeight: 600, color: 'primary.main', mt: 1 }}>
              {recognizedPerson.message}
            </Typography>
          )}
        </Alert>
      )}

      {showForm && (
        <Box component="form" onSubmit={handleSubmit} noValidate sx={{ mt: 2 }}>
          <Typography variant="subtitle1" fontWeight="bold" gutterBottom>
            New Person Detected - Please Enter Details:
          </Typography>
          <Grid container spacing={2}>
            <Grid item xs={12} sm={6}>
              <TextField
                label="First Name *"
                name="firstName"
                fullWidth
                value={formData.firstName}
                onChange={handleInputChange}
                required
              />
            </Grid>
            <Grid item xs={12} sm={6}>
              <TextField
                label="Last Name *"
                name="lastName"
                fullWidth
                value={formData.lastName}
                onChange={handleInputChange}
                required
              />
            </Grid>
            <Grid item xs={12} sm={4}>
              <TextField
                label="Age"
                type="number"
                name="age"
                fullWidth
                value={formData.age}
                onChange={handleInputChange}
              />
            </Grid>
            <Grid item xs={12} sm={4}>
              <TextField
                label="Height (cm)"
                name="height"
                fullWidth
                value={formData.height}
                onChange={handleInputChange}
              />
            </Grid>
            <Grid item xs={12} sm={4}>
              <TextField
                label="Weight (kg)"
                name="weight"
                fullWidth
                value={formData.weight}
                onChange={handleInputChange}
              />
            </Grid>
            <Grid item xs={12}>
              <Button
                type="submit"
                variant="contained"
                fullWidth
                color="success"
                disabled={loading}
              >
                {loading ? 'Processing...' : 'Add Person'}
              </Button>
            </Grid>
          </Grid>
        </Box>
      )}
    </Paper>
  );
};

export default FaceDetectionSystem;
