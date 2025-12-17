import {useNavigate} from "react-router-dom";

export const BackArrow = () => {
    const navigate = useNavigate();


    return (
        <>
            <span onClick={() => navigate(-1)} className="cursor-pointer">
                <h1 className="text-4xl">&#8592;</h1>
            </span>
        </>
    );
};

export default BackArrow;