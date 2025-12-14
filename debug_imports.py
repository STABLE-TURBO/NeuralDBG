try:
    import neural
    print("neural imported")
    import neural.exceptions
    print("neural.exceptions imported")
    import neural.aquarium
    print("neural.aquarium imported")
    import neural.dashboard
    print("neural.dashboard imported")
except ImportError as e:
    print(f"ImportError: {e}")
except Exception as e:
    print(f"Error: {e}")
