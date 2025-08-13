    
    def cleanup_models(self):
        """Public method to clean up all models and free memory"""
        logger.info("Cleaning up WAN models to free memory...")
        self._cleanup_memory()
    
    def get_current_model(self) -> Optional[str]:
        """Get the currently loaded model type"""
        return self.current_model
