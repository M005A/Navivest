import pyzed.sl as sl

def main():
    
    zed = sl.Camera()
    
    init_params = sl.InitParameters()
    init_params.sdk_verbose = 0
    
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        exit(1)
        
    
    zed_serial = zed.get_camera_information().serial_number
    print("Hello! This is my serial number: {0}".format(zed_serial))
    
    zed.close()
    

if __name__ == "__main__":
    main()