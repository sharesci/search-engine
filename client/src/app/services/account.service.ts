import { Injectable } from '@angular/core'
import { Http, Response, Headers, RequestOptions, URLSearchParams} from '@angular/http'
import { Observable } from 'rxjs';
import { AppConfig } from '../app.config.js';
import 'rxjs/add/operator/map';

@Injectable()
export class AccountService {
    
    constructor(private _http: Http, private _config: AppConfig) { }
    
    private _createAccountUrl = this._config.apiUrl + "/account/create";

    create(username: string, password: string): Observable<any> {
        let data = new URLSearchParams();
        data.append('username', username);
        data.append('password', password);
        
        return this._http.post(this._createAccountUrl, data)
                .map((response: Response) => response.json());
    }
}